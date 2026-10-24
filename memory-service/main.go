package main

import (
	"context"
	_ "embed"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed schema.sql
var schemaSQL string

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func connect(ctx context.Context, url string) *pgxpool.Pool {
	var lastErr error
	for i := 0; i < 30; i++ {
		pool, err := pgxpool.New(ctx, url)
		if err == nil {
			if err = pool.Ping(ctx); err == nil {
				return pool
			}
			pool.Close()
		}
		lastErr = err
		log.Printf("waiting for postgres: %v", err)
		time.Sleep(time.Second)
	}
	log.Fatalf("postgres unavailable: %v", lastErr)
	return nil
}

func main() {
	ctx := context.Background()
	pool := connect(ctx, env("DATABASE_URL", "postgres://skillforge:skillforge@localhost:5432/skillforge"))
	defer pool.Close()
	if _, err := pool.Exec(ctx, schemaSQL); err != nil {
		log.Fatalf("migrate: %v", err)
	}
	s := &Server{db: pool}

	if secs, _ := strconv.Atoi(env("REAP_INTERVAL_SECONDS", "0")); secs > 0 {
		go func() {
			for range time.Tick(time.Duration(secs) * time.Second) {
				if _, err := s.reap(ctx, nowOr(nil), nil); err != nil {
					log.Printf("reap: %v", err)
				}
			}
		}()
	}
	addr := env("LISTEN_ADDR", ":8080")
	log.Printf("memory-service listening on %s", addr)
	srv := &http.Server{Addr: addr, Handler: s.routes(), ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
