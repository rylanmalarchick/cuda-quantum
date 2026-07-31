/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/RuntimeEndpoint.h"
#include <nanobind/nanobind.h>

namespace cudaq {

/// Build a `RuntimeEndpoint` that dispatches kernel launches to \p obj, a
/// Python object implementing one or more of the protocols declared in
/// `python/cudaq/_experimental/runtime_endpoint.py`.
///
/// Only the policies with a `python::py_launch_protocol` specialization can be
/// dispatched; the rest keep a null entry so that `launchKernel` reports the
/// policy as unsupported. Whether \p obj actually implements the corresponding
/// Python method is checked at launch time, so one object may support any
/// subset of the protocols.
RuntimeEndpoint makePyRuntimeEndpoint(nanobind::object obj);

/// Create python bindings for C++ code in this compilation unit.
void bindRuntimeEndpoint(nanobind::module_ &mod);

} // namespace cudaq
