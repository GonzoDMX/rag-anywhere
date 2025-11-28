package models

import (
	"bytes"
	"encoding/binary"
	"math"
)

// Convert vector embeddings to/from byte slices for BLOB storage in SQLite

// Float32ToBytes converts a slice of float32 to a byte slice for BLOB storage
func Float32ToBytes(floats []float32) []byte {
	buf := new(bytes.Buffer)
	for _, f := range floats {
		err := binary.Write(buf, binary.LittleEndian, f)
		if err != nil {
			return nil
		}
	}
	return buf.Bytes()
}

// BytesToFloat32 converts a BLOB back to a slice of float32
func BytesToFloat32(data []byte) []float32 {
	if len(data)%4 != 0 {
		return nil
	}
	floats := make([]float32, len(data)/4)
	for i := 0; i < len(floats); i++ {
		bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}
