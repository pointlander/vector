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
	"github.com/pointlander/matrix/vector"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
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

func mark1() {
	const (
		N = 7 * 11
	)
	n := N / 2
	rng := rand.New(rand.NewSource(1))
	x := NewZeroMatrix(4, 1)
	y := NewZeroMatrix(4, 1)
	set := make([]int, 1024)
	for i := range set {
		set[i] = rng.Intn(n) + 1
	}
	samples := make([]Line, 256)
	for i := range samples {
		for j := range x.Data {
			a := set[rng.Intn(len(set))]
			x.Data[j] = float32(a)
			y.Data[j] = float32(N % a)
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
			if !math.IsNaN(x) && !math.IsNaN(y) {
				points = append(points, Point{
					X: x,
					Y: y,
					A: samples[i],
					B: samples[j],
				})
			}
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
		points[i].Cost /= float64(len(points))
	}
	sort.Slice(points, func(i, j int) bool {
		return points[i].Cost < points[j].Cost
	})
	fmt.Println()
	for i := range points[:10] {
		fmt.Println(points[i].X, points[i].Y, points[i].Cost)
	}
	fmt.Println(N, points[0].X*points[0].Y, n)
}

func softmax(values []float32) {
	max := float32(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := float32(0.0)
	for j, value := range values {
		values[j] = float32(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

func SelfAttentionX(Q, K, V Matrix) Matrix {
	o := Matrix{
		Cols: V.Cols,
		Rows: K.Rows,
		Data: make([]float32, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]float32, V.Cols), make([]float32, Q.Rows)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = vector.Dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = vector.Dot(values, V)
		}
		//softmax(outputs)
		o.Data = append(o.Data, outputs...)
	}
	return o
}

func main() {
	rng := rand.New(rand.NewSource(1))
	q := NewRandomMatrix(4, 4)
	k := NewRandomMatrix(4, 4)
	v := NewRandomMatrix(4, 4)
	values := make(plotter.Values, 0, 1024)
	for i := 0; i < 128; i++ {
		x := SelfAttentionX(q.Sample(rng), k.Sample(rng), v.Sample(rng))
		for _, value := range x.Data {
			values = append(values, float64(value))
		}
	}
	p := plot.New()
	p.Title.Text = "distribution"
	histogram, err := plotter.NewHist(values, 256)
	if err != nil {
		panic(err)
	}
	p.Add(histogram)
	err = p.Save(8*vg.Inch, 8*vg.Inch, "distribution.png")
	if err != nil {
		panic(err)
	}
}
