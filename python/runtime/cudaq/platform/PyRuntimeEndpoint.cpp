/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PyRuntimeEndpoint.h"
#include "common/KernelArgs.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "cudaq/algorithms/observe/policy.h"
#include "cudaq/algorithms/policies.h"
#include "cudaq/algorithms/run/policy.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <memory>
#include <nanobind/stl/string.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/// Owns the Python endpoint object on behalf of the `RuntimeEndpoint`.
///
/// `RuntimeEndpoint::impl` is a `std::any`, which requires a copy-constructible
/// payload, and copying a `nanobind::object` mutates a refcount and therefore
/// needs the GIL. Holding this through a `std::shared_ptr` keeps the `std::any`
/// copies GIL-free and confines the refcount work to construction and teardown.
struct PyEndpointObject {
  nanobind::object obj;

  explicit PyEndpointObject(nanobind::object obj) : obj(std::move(obj)) {}

  PyEndpointObject(const PyEndpointObject &) = delete;
  PyEndpointObject &operator=(const PyEndpointObject &) = delete;

  ~PyEndpointObject() {
    if (!obj.is_valid())
      return;

    // The endpoint can outlive the interpreter, in which case there is no
    // refcount left to decrement. Leak the handle rather than touch freed
    // interpreter state.
    if (!Py_IsInitialized()) {
      (void)obj.release();
      return;
    }

    nanobind::gil_scoped_acquire gil;
    obj = nanobind::object();
  }
};

using EndpointHandle = std::shared_ptr<PyEndpointObject>;

/// The `cudaq::KernelArgs` view handed to a Python endpoint.
///
/// Unlike `KernelArgs`, this does not support packed arguments (a C++-only
/// path). It carries the argument types recovered from the compiled module so
/// that scalar arguments can be converted back to Python values.
struct PyKernelArgs {
  std::span<void *const> args;
  std::optional<std::vector<mlir::Type>> argTypes;
  mlir::ModuleOp module;

  PyKernelArgs() = default;

  PyKernelArgs(cudaq::KernelArgs kargs, const cudaq::CompiledModule &compiled) {
    if (kargs.hasPacked())
      CUDAQ_WARN("Packed arguments are not supported for Python runtime "
                 "endpoints. Ignoring arguments.");
    if (auto typeErasedArgs = kargs.getTypeErased())
      args = typeErasedArgs.value();
    parseSignature(compiled);
  }

private:
  /// Recover the kernel argument types from the compiled module's MLIR.
  void parseSignature(const cudaq::CompiledModule &compiled) {
    auto mlirArt = compiled.getMlir();
    if (!mlirArt)
      return;
    mlir::ModuleOp modOp =
        cudaq_internal::compiler::CompiledModuleHelper::getMlirModuleOp(
            *mlirArt);
    if (!modOp)
      return;
    auto funcOp =
        cudaq::getKernelFuncOp</*noThrow=*/true>(modOp, compiled.getName());
    if (!funcOp)
      return;
    auto inputs = funcOp.getFunctionType().getInputs();
    module = modOp;
    argTypes = std::vector<mlir::Type>(inputs.begin(), inputs.end());
  }
};

/// Convert a single kernel argument to a Python object.
nanobind::object convertArg(mlir::ModuleOp module, mlir::Type ty, void *data) {
  if (!data)
    return nanobind::none();
  if (!mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::ComplexType>(ty))
    throw std::runtime_error("unsupported argument type: " +
                             cudaq::mlirTypeToString(ty));
  return cudaq::convertResult(module, ty, reinterpret_cast<char *>(data));
}

std::string reprKernelArgs(const PyKernelArgs &self) {
  if (!self.argTypes)
    return "KernelArgs(<unknown args>)";

  std::string out = "KernelArgs([";
  for (std::size_t i = 0; i < self.argTypes->size(); ++i) {
    if (i)
      out += ", ";
    mlir::Type ty = (*self.argTypes)[i];
    void *data = i < self.args.size() ? self.args[i] : nullptr;
    try {
      nanobind::object obj = convertArg(self.module, ty, data);
      out += nanobind::cast<std::string>(
          nanobind::steal<nanobind::object>(PyObject_Repr(obj.ptr())));
    } catch (const std::exception &) {
      out += "<instance of " + cudaq::mlirTypeToString(ty) + ">";
    }
  }
  out += "])";
  return out;
}

} // namespace

namespace cudaq::python {

/// Maps a launch policy onto the Python endpoint protocol.
///
/// Specialize this to make a policy reachable from Python. A specialization
/// provides
///   - `method`: the name of the Python method implementing the protocol,
///   - `kwargs(policy)`: the policy-derived keyword arguments passed to it,
///   - optionally `fromPython(object)`: converts the returned Python object to
///     `Policy::result_type`. Defaults to a plain `nanobind::cast`, which is
///     enough for every result type that has bindings of its own.
///
/// The primary template is intentionally empty rather than undefined, so that
/// probing it for `method` is a clean substitution failure: a policy without a
/// specialization cannot be dispatched and keeps a null entry in the dispatch
/// table.
template <typename Policy>
struct py_launch_protocol {};

template <>
struct py_launch_protocol<sample_policy> {
  static constexpr const char *method = "sample";

  static nanobind::dict kwargs(const sample_policy &policy) {
    nanobind::dict kw;
    kw["shots_count"] = policy.options.shots;
    kw["explicit_measurements"] = policy.options.explicit_measurements;
    return kw;
  }
};

template <>
struct py_launch_protocol<observe_policy> {
  static constexpr const char *method = "observe";

  static nanobind::dict kwargs(const observe_policy &policy) {
    nanobind::dict kw;
    // An observe launch may carry no shot count (-1 means "not applicable").
    if (policy.options.shots >= 0)
      kw["shots_count"] = policy.options.shots;
    // Hand over an owned copy: the policy is only alive for this launch.
    kw["spin_operator"] =
        nanobind::cast(policy.spin, nanobind::rv_policy::copy);
    return kw;
  }
};

template <>
struct py_launch_protocol<run_policy> {
  static constexpr const char *method = "run";

  static nanobind::dict kwargs(const run_policy &policy) {
    nanobind::dict kw;
    kw["shots_count"] = policy.shots;
    return kw;
  }

  /// `run_result` has no bindings of its own; the protocol returns the QIR
  /// output log directly.
  static run_result fromPython(nanobind::object result) {
    return run_result{nanobind::cast<std::string>(result)};
  }
};

/// Satisfied by the policies that have a Python protocol mapping.
template <typename Policy>
concept py_launchable = requires { py_launch_protocol<Policy>::method; };

namespace {

/// Dispatch a launch of \p policy to the Python object held in \p impl.
template <py_launchable Policy>
typename Policy::result_type pyLaunch(std::any &impl, const Policy &policy,
                                      const CompiledModule &module,
                                      KernelArgs args) {
  using protocol = py_launch_protocol<Policy>;

  // `cudaq.run` drops the GIL around the launch while `sample` and `observe`
  // hold it; the guard handles both.
  nanobind::gil_scoped_acquire gil;

  auto &obj = std::any_cast<EndpointHandle &>(impl)->obj;
  if (!nanobind::hasattr(obj, protocol::method))
    throw std::runtime_error(
        std::string("The runtime endpoint does not implement '") +
        protocol::method + "', which is required to launch the '" +
        get_policy_name(policy) + "' policy.");

  CUDAQ_INFO("Dispatching a '{}' launch to the Python runtime endpoint.",
             get_policy_name(policy));

  // Hand Python its own copy of the module so that an endpoint holding on to it
  // does not outlive the launch. The argument span cannot be copied this way —
  // it stays owned by the caller and is only valid for the duration of the
  // call, which the protocol documents.
  auto pyModule =
      nanobind::cast(CompiledModule(module), nanobind::rv_policy::move);
  auto pyArgs = nanobind::cast(PyKernelArgs(std::move(args), module),
                               nanobind::rv_policy::move);

  auto kwargs = protocol::kwargs(policy);
  auto result = obj.attr(protocol::method)(pyModule, pyArgs, **kwargs);

  if constexpr (requires { protocol::fromPython(result); })
    return protocol::fromPython(std::move(result));
  else
    return nanobind::cast<typename Policy::result_type>(result);
}

} // namespace
} // namespace cudaq::python

cudaq::RuntimeEndpoint cudaq::makePyRuntimeEndpoint(nanobind::object obj) {
  using namespace cudaq::python;

  RuntimeEndpoint endpoint;
  endpoint.dispatch =
      detail::DispatchTable<all_policies>::create([]<typename Policy>() {
        if constexpr (py_launchable<Policy>)
          return &pyLaunch<Policy>;
        else
          return static_cast<detail::launch_fn_type<Policy>>(nullptr);
      });
  endpoint.impl = std::make_shared<PyEndpointObject>(std::move(obj));
  return endpoint;
}

void cudaq::bindRuntimeEndpoint(nanobind::module_ &mod) {
  nanobind::class_<PyKernelArgs>(mod, "KernelArgs",
                                 "The processed arguments of a kernel launch.")
      .def(
          "__len__",
          [](const PyKernelArgs &self) -> std::size_t {
            if (!self.argTypes)
              throw std::runtime_error("unknown number of arguments");
            return self.argTypes->size();
          },
          "The number of kernel arguments. Requires a known signature.")
      .def(
          "__getitem__",
          [](const PyKernelArgs &self, std::size_t i) -> nanobind::object {
            if (!self.argTypes)
              throw std::runtime_error("unknown argument type");
            if (i >= self.argTypes->size() || i >= self.args.size())
              throw nanobind::index_error();
            return convertArg(self.module, (*self.argTypes)[i], self.args[i]);
          },
          "Convert the argument at the given index to a Python value. Requires "
          "a known signature and a supported (scalar) argument type.")
      .def("__repr__", &reprKernelArgs);

  mod.def(
      "set_runtime_endpoint",
      [](nanobind::object endpoint, std::size_t qpu_id) {
        get_platform().setRuntimeEndpoint(
            makePyRuntimeEndpoint(std::move(endpoint)), qpu_id);
      },
      nanobind::arg("endpoint"), nanobind::arg("qpu_id") = 0,
      "Route kernel launches on the given QPU to a Python object implementing "
      "one or more of the cudaq._experimental runtime endpoint protocols.");
}
