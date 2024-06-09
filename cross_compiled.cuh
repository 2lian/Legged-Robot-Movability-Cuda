#pragma once
#include "HeaderCUDA.h"

template <typename T_in, typename param, typename T_out>
void apply_kernel(const Array<T_in> input, const param dim,
                  void (*kernel)(const Array<T_in>, const param,
                                 Array<T_out> const),
                  Array<T_out> const output);
