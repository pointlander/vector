// Copyright 2024 The Vector Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	. "github.com/pointlander/matrix"
)

// Line is a line sample
type Line struct {
	B1, B0 float64
	Cost   float64
}

func main() {
	rng := rand.New(rand.NewSource(1))
	x := NewZeroMatrix(10, 1)
	y := NewZeroMatrix(10, 1)
	samples := make([]Line, 10)
	for i := range samples {
		for j := range x.Data {
			a := rng.Intn(100) + 1
			x.Data[j] = float32(a)
			y.Data[j] = float32(77 % a)
		}
		b0, b1 := LinearRegression(x, y)
		samples[i].B0 = b0
		samples[i].B1 = b1
	}
	for i := range samples {
		fmt.Println(samples[i])
	}
}
