package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	hf "github.com/hupe1980/go-huggingface"
	"github.com/joho/godotenv"
)

type AIModelConnector struct {
	Client *http.Client
}

type Inputs struct {
	Table map[string][]string `json:"table"`
	Query string              `json:"query"`
}

type Response struct {
	Answer      string   `json:"answer"`
	Coordinates [][]int  `json:"coordinates"`
	Cells       []string `json:"cells"`
	Aggregator  string   `json:"aggregator"`
}

func CsvToSlice(data string) (map[string][]string, error) {
	// Membuat pembaca CSV dari string data yang diberikan
	reader := csv.NewReader(strings.NewReader(data))

	// Membaca semua baris dari data CSV
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	// Inisialisasi peta hasil dengan kunci string dan nilai slice string
	result := make(map[string][]string)
	if len(lines) == 0 {
		// Jika tidak ada baris dalam data CSV, kembalikan peta kosong
		return result, nil
	}

	// Mengambil baris pertama sebagai header
	headers := lines[0]
	for _, header := range headers {
		// Inisialisasi setiap header dengan slice kosong dalam peta hasil
		result[header] = []string{}
	}

	// Iterasi melalui baris data (mengabaikan baris pertama yang merupakan header)
	for _, line := range lines[1:] {
		for i, value := range line {
			// Menambahkan nilai ke dalam slice yang sesuai dengan header
			result[headers[i]] = append(result[headers[i]], value)
		}
	}

	// Mengembalikan hasil peta dan nil (tidak ada error)
	return result, nil
}

func (c *AIModelConnector) ConnectAIModel(payload interface{}, token string) (Response, error) {
	// Coba konversi payload ke tipe Inputs
	inputs, ok := payload.(Inputs)
	if !ok {
		// Jika gagal, kembalikan error karena tipe payload tidak valid
		return Response{}, errors.New("invalid payload type")
	}

	// Serialize inputs menjadi JSON
	reqBody, err := json.Marshal(inputs)
	if err != nil {
		// Jika terjadi error saat serialisasi, kembalikan error
		return Response{}, err
	}

	// Buat permintaan HTTP POST ke URL API
	req, err := http.NewRequest("POST", "https://api-inference.huggingface.co/models/openai-community/gpt2", bytes.NewBuffer(reqBody))
	if err != nil {
		// Jika terjadi error saat membuat permintaan, kembalikan error
		return Response{}, err
	}

	// Set header Authorization dengan token yang diberikan
	req.Header.Set("Authorization", "Bearer "+token)
	// Set header Content-Type sebagai application/json
	req.Header.Set("Content-Type", "application/json")

	// Kirim permintaan HTTP menggunakan client
	resp, err := c.Client.Do(req)
	if err != nil {
		// Jika terjadi error saat mengirim permintaan, kembalikan error
		return Response{}, err
	}
	// Pastikan untuk menutup body respons setelah selesai
	defer resp.Body.Close()

	// Periksa status kode respons, jika tidak OK, kembalikan error
	if resp.StatusCode != http.StatusOK {
		return Response{}, fmt.Errorf("failed to connect to AI model with status: %d", resp.StatusCode)
	}

	// Decode body respons JSON ke dalam struct Response
	var result Response
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		// Jika terjadi error saat decoding, kembalikan error
		return Response{}, err
	}

	// Kembalikan hasil decoding sebagai Response dan nil untuk error
	return result, nil
}

func main() {
	// Buka file CSV dengan nama "data-series.csv"
	file, err := os.Open("data-series.csv")
	if err != nil {
		// Jika terjadi error saat membuka file, log error dan hentikan program
		log.Fatalf("Failed to open file: %v", err)
	}
	// Pastikan file ditutup setelah selesai digunakan
	defer file.Close()

	// Buat pembaca CSV untuk membaca file
	reader := csv.NewReader(file)
	// Baca semua baris dari file CSV
	lines, err := reader.ReadAll()
	if err != nil {
		// Jika terjadi error saat membaca data CSV, log error dan hentikan program
		log.Fatalf("Failed to read CSV data: %v", err)
	}

	// Gunakan strings.Builder untuk menggabungkan baris CSV menjadi string
	var data strings.Builder
	for _, line := range lines {
		// Gabungkan setiap baris menjadi string dengan pemisah koma dan tambahkan newline di akhir
		data.WriteString(strings.Join(line, ",") + "\n")
	}

	// Konversi data CSV menjadi string
	csvData := data.String()
	// Panggil fungsi CsvToSlice untuk mengkonversi string CSV menjadi peta
	result, err := CsvToSlice(csvData)
	if err != nil {
		// Jika terjadi error saat konversi CSV, log error dan hentikan program
		log.Fatalf("Failed to convert CSV to slice: %v", err)
	}

	// Load variabel lingkungan dari file .env
	if err := godotenv.Load(); err != nil {
		// Jika terjadi error saat memuat .env, log error dan hentikan program
		log.Fatalf("Error loading .env file: %v", err)
	}

	// Dapatkan nilai token dari variabel lingkungan
	token := os.Getenv("HUGGINGFACE_TOKEN")
	if token == "" {
		// Jika token tidak diset di .env, log error dan hentikan program
		log.Fatal("HUGGINGFACE_TOKEN is required but not set in .env")
	}

	// Buat klien inference baru menggunakan token yang diberikan
	ic := hf.NewInferenceClient(token)

	// Ambil input query dari pengguna
	var query string
	fmt.Print("Can I Help You ? : ")
	fmt.Scanln(&query)

	// Buat struct Inputs dengan data tabel dan query
	article := Inputs{
		Table: result,
		Query: query,
	}

	// Konversi struct Inputs menjadi JSON
	articleJSON, err := json.Marshal(article)
	if err != nil {
		// Jika terjadi error saat mengkonversi ke JSON, log error dan hentikan program
		log.Fatalf("Error marshaling article to JSON: %v", err)
	}

	// Panggil metode summarization dari klien inference dengan artikel yang telah di-JSON-kan
	summary, err := ic.Summarization(context.Background(), &hf.SummarizationRequest{
		Inputs: []string{string(articleJSON)},
	})
	if err != nil {
		// Jika terjadi error saat melakukan summarization, log error dan hentikan program
		log.Fatalf("Error summarizing text: %v", err)
	}

	// Cetak teks ringkasan pertama yang dikembalikan oleh API
	fmt.Println(summary[0].SummaryText)
}
