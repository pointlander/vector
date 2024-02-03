// Copyright 2024 The Vector Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	. "github.com/pointlander/matrix"
)

// Line is a line sample
type Line struct {
	B1, B0 float64
}

// Point is a point
type Point struct {
	X    float64
	Y    float64
	A    Line
	B    Line
	Cost float64
}

func main() {
	rng := rand.New(rand.NewSource(1))
	x := NewZeroMatrix(32, 1)
	y := NewZeroMatrix(32, 1)
	samples := make([]Line, 300)
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
	fmt.Println()
	points := make([]Point, 0, 8)
	for i := 0; i < len(samples); i++ {
		for j := i + 1; j < len(samples); j++ {
			x := (samples[j].B0 - samples[i].B0) / (samples[i].B1 - samples[j].B1)
			y := samples[i].B1*x + samples[i].B0
			points = append(points, Point{
				X: x,
				Y: y,
				A: samples[i],
				B: samples[j],
			})
		}
	}
	for i := range points {
		fmt.Println(points[i])
	}
	for i := 0; i < len(points); i++ {
		for j := 0; j < len(points); j++ {
			diffx := points[i].X - points[j].X
			diffy := points[i].Y - points[j].Y
			points[i].Cost += math.Sqrt(diffx*diffx + diffy*diffy)
		}
	}
	sort.Slice(points, func(i, j int) bool {
		return points[i].Cost < points[j].Cost
	})
	fmt.Println()
	for i := range points[:10] {
		fmt.Println(points[i].X, points[i].Y, points[i].Cost)
	}
}
