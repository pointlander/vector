// Copyright 2024 The Vector Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	. "github.com/pointlander/matrix"
)

func main() {
	x := NewMatrix(10, 1, []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}...)
	y := NewMatrix(10, 1, []float32{1, 3, 2, 5, 7, 8, 8, 9, 10, 12}...)
	b0, b1 := LinearRegression(x, y)
	fmt.Println(b0, b1)
}
