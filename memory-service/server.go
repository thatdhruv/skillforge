package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Server struct {
	db *pgxpool.Pool
}

// ---------------------------------------------------------------- helpers

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func fail(w http.ResponseWriter, code int, err error) {
	writeJSON(w, code, map[string]string{"error": err.Error()})
}

func nowOr(p *float64) float64 {
	if p != nil {
		return *p
	}
	return float64(time.Now().UnixNano()) / 1e9
}

// vecLit renders an embedding as a pgvector literal; nil means SQL NULL.
func vecLit(e []float64) *string {
	if len(e) == 0 {
		return nil
	}
	parts := make([]string, len(e))
	for i, x := range e {
		parts[i] = fmt.Sprintf("%g", x)
	}
	s := "[" + strings.Join(parts, ",") + "]"
	return &s
}

var tokRe = regexp.MustCompile(`[a-z0-9_]+`)

// orQuery turns free text into a tsquery that ORs its tokens (recall first; ranking sorts it out).
func orQuery(q string) string {
	toks := tokRe.FindAllString(strings.ToLower(q), -1)
	if len(toks) == 0 {
		return "''"
	}
	return strings.Join(toks, " | ")
}

// hybrid runs vector and full-text retrieval over a table and fuses them with RRF.
// where must reference only $1..$n of args; the query/embedding are appended after them.
func (s *Server) hybrid(ctx context.Context, table, where string, args []any, q string, emb []float64, k int, mode string) ([]map[string]any, error) {
	n := len(args)
	var vecIDs, kwIDs []int64
	if lit := vecLit(emb); lit != nil && mode != "keyword" {
		rows, err := s.db.Query(ctx, fmt.Sprintf(
			`SELECT id, 1 - (embedding <=> $%d::vector) FROM %s WHERE %s AND embedding IS NOT NULL
			 ORDER BY embedding <=> $%d::vector LIMIT 50`,
			n+1, table, where, n+1), append(append([]any{}, args...), *lit)...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var id int64
			var sim float64
			if err := rows.Scan(&id, &sim); err != nil {
				return nil, err
			}
			if sim >= minSimilarity { // nearest neighbours of an unrelated query are noise, not evidence
				vecIDs = append(vecIDs, id)
			}
		}
		rows.Close()
	}
	if mode == "vector" {
		return s.fuse(ctx, table, vecIDs, nil, k)
	}
	rows, err := s.db.Query(ctx, fmt.Sprintf(
		`SELECT id FROM %s WHERE %s AND tsv @@ to_tsquery('english', $%d)
		 ORDER BY ts_rank_cd(tsv, to_tsquery('english', $%d)) DESC LIMIT 50`,
		table, where, n+1, n+1), append(append([]any{}, args...), orQuery(q))...)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		kwIDs = append(kwIDs, id)
	}
	rows.Close()

	return s.fuse(ctx, table, vecIDs, kwIDs, k)
}

func (s *Server) fuse(ctx context.Context, table string, vecIDs, kwIDs []int64, k int) ([]map[string]any, error) {
	order, score := rrf(vecIDs, kwIDs)
	if len(order) > k {
		order = order[:k]
	}
	if len(order) == 0 {
		return []map[string]any{}, nil
	}
	rs, err := s.db.Query(ctx, fmt.Sprintf(
		`SELECT id, to_jsonb(t) - 'embedding' - 'tsv' FROM %s t WHERE id = ANY($1)`, table), order)
	if err != nil {
		return nil, err
	}
	defer rs.Close()
	byID := map[int64]map[string]any{}
	for rs.Next() {
		var id int64
		var raw []byte
		if err := rs.Scan(&id, &raw); err != nil {
			return nil, err
		}
		m := map[string]any{}
		_ = json.Unmarshal(raw, &m)
		m["score"] = score[id]
		byID[id] = m
	}
	out := make([]map[string]any, 0, len(order))
	for _, id := range order {
		if m, ok := byID[id]; ok {
			out = append(out, m)
		}
	}
	return out, nil
}

func (s *Server) event(ctx context.Context, tx pgx.Tx, ts float64, typ, name string, version int, success *bool) error {
	_, err := tx.Exec(ctx, `INSERT INTO skill_events(ts,type,name,version,success) VALUES($1,$2,$3,$4,$5)`,
		ts, typ, name, version, success)
	return err
}

// ---------------------------------------------------------------- routes

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		if err := s.db.Ping(r.Context()); err != nil {
			fail(w, 503, err)
			return
		}
		writeJSON(w, 200, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("POST /v1/episodes", s.addEpisode)
	mux.HandleFunc("POST /v1/episodes/search", s.searchEpisodes)
	mux.HandleFunc("POST /v1/facts", s.putFact)
	mux.HandleFunc("POST /v1/facts/search", s.searchFacts)
	mux.HandleFunc("POST /v1/skills", s.putSkill)
	mux.HandleFunc("POST /v1/skills/search", s.searchSkills)
	mux.HandleFunc("GET /v1/skills", s.listSkills)
	mux.HandleFunc("GET /v1/skills/events", s.listEvents)
	mux.HandleFunc("POST /v1/skills/{name}/usage", s.recordUsage)
	mux.HandleFunc("POST /v1/admin/reap", s.reapHandler)
	mux.HandleFunc("GET /v1/stats", s.stats)
	if os.Getenv("ALLOW_RESET") == "1" { // test/eval convenience, never enabled in the Helm chart
		mux.HandleFunc("POST /v1/admin/reset", func(w http.ResponseWriter, r *http.Request) {
			_, err := s.db.Exec(r.Context(), `TRUNCATE episodes, facts, skills, skill_events RESTART IDENTITY`)
			if err != nil {
				fail(w, 500, err)
				return
			}
			writeJSON(w, 200, map[string]string{"status": "reset"})
		})
	}
	return mux
}

// ---------------------------------------------------------------- episodic

type episodeIn struct {
	ID         string          `json:"id"`
	TaskID     string          `json:"task_id"`
	Kind       string          `json:"kind"`
	Success    bool            `json:"success"`
	Prompt     string          `json:"prompt"`
	Answer     string          `json:"answer"`
	TokensIn   int             `json:"tokens_in"`
	TokensOut  int             `json:"tokens_out"`
	Cost       float64         `json:"cost"`
	Trajectory json.RawMessage `json:"trajectory"`
	Meta       json.RawMessage `json:"meta"`
	Embedding  []float64       `json:"embedding"`
	CreatedAt  *float64        `json:"created_at"`
}

func jsonOrNull(raw json.RawMessage) any {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	return string(raw)
}

func (s *Server) addEpisode(w http.ResponseWriter, r *http.Request) {
	var in episodeIn
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		fail(w, 400, err)
		return
	}
	if in.ID == "" {
		in.ID = fmt.Sprintf("ep-%d", time.Now().UnixNano())
	}
	_, err := s.db.Exec(r.Context(), `INSERT INTO episodes
		(id,task_id,kind,success,prompt,answer,tokens_in,tokens_out,cost,trajectory,meta,embedding,created_at)
		VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11::jsonb,$12::vector,$13)
		ON CONFLICT (id) DO NOTHING`,
		in.ID, in.TaskID, in.Kind, in.Success, in.Prompt, in.Answer, in.TokensIn, in.TokensOut, in.Cost,
		jsonOrNull(in.Trajectory), jsonOrNull(in.Meta), vecLit(in.Embedding), nowOr(in.CreatedAt))
	if err != nil {
		fail(w, 500, err)
		return
	}
	writeJSON(w, 201, map[string]string{"id": in.ID})
}

func (s *Server) searchEpisodes(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Embedding   []float64 `json:"embedding"`
		K           int       `json:"k"`
		OnlySuccess bool      `json:"only_success"`
	}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		fail(w, 400, err)
		return
	}
	if in.K <= 0 {
		in.K = 10
	}
	rows, err := s.db.Query(r.Context(), `SELECT to_jsonb(t) - 'embedding' - 'trajectory' ||
		jsonb_build_object('score', 1 - (embedding <=> $1::vector))
		FROM episodes t WHERE embedding IS NOT NULL AND ($3 = false OR success)
		ORDER BY embedding <=> $1::vector LIMIT $2`, vecLit(in.Embedding), in.K, in.OnlySuccess)
	if err != nil {
		fail(w, 500, err)
		return
	}
	defer rows.Close()
	out := []json.RawMessage{}
	for rows.Next() {
		var raw []byte
		if err := rows.Scan(&raw); err != nil {
			fail(w, 500, err)
			return
		}
		out = append(out, raw)
	}
	writeJSON(w, 200, out)
}

// ---------------------------------------------------------------- semantic

type factIn struct {
	Subject       string    `json:"subject"`
	Predicate     string    `json:"predicate"`
	Object        string    `json:"object"`
	Text          string    `json:"text"`
	Embedding     []float64 `json:"embedding"`
	Confidence    *float64  `json:"confidence"`
	ValidFrom     *float64  `json:"valid_from"`
	TTLSeconds    *float64  `json:"ttl_seconds"`
	SourceEpisode *string   `json:"source_episode"`
	Now           *float64  `json:"now"`
}

func (s *Server) putFact(w http.ResponseWriter, r *http.Request) {
	var in factIn
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		fail(w, 400, err)
		return
	}
	if in.Subject == "" || in.Predicate == "" || in.Object == "" {
		fail(w, 400, fmt.Errorf("subject, predicate and object are required"))
		return
	}
	now := nowOr(in.Now)
	vf := now
	if in.ValidFrom != nil {
		vf = *in.ValidFrom
	}
	conf := 0.8
	if in.Confidence != nil {
		conf = *in.Confidence
	}
	var exp *float64
	if in.TTLSeconds != nil {
		e := vf + *in.TTLSeconds
		exp = &e
	}
	text := in.Text
	if text == "" {
		text = in.Subject + " " + in.Predicate + " " + in.Object
	}
	ctx := r.Context()
	tx, err := s.db.Begin(ctx)
	if err != nil {
		fail(w, 500, err)
		return
	}
	defer tx.Rollback(ctx)

	// expire stale incumbents for this key first so the one-active-per-key index never blocks a new fact
	if _, err = tx.Exec(ctx, `UPDATE facts SET status='expired' WHERE subject=$1 AND predicate=$2
		AND status='active' AND expires_at IS NOT NULL AND expires_at <= $3`, in.Subject, in.Predicate, now); err != nil {
		fail(w, 500, err)
		return
	}
	var curID int64
	var cur factRec
	var curPtr *factRec
	err = tx.QueryRow(ctx, `SELECT id, object, valid_from, confidence, expires_at FROM facts
		WHERE subject=$1 AND predicate=$2 AND status='active' FOR UPDATE`, in.Subject, in.Predicate).
		Scan(&curID, &cur.Object, &cur.ValidFrom, &cur.Confidence, &cur.ExpiresAt)
	if err == nil {
		curPtr = &cur
	} else if err != pgx.ErrNoRows {
		fail(w, 500, err)
		return
	}
	incoming := factRec{Object: in.Object, ValidFrom: vf, Confidence: conf, ExpiresAt: exp}
	action := decideFact(curPtr, incoming, now)

	insert := func(status string) (int64, error) {
		var id int64
		err := tx.QueryRow(ctx, `INSERT INTO facts(subject,predicate,object,text,embedding,confidence,valid_from,
			expires_at,last_confirmed,status,source_episode) VALUES($1,$2,$3,$4,$5::vector,$6,$7,$8,$9,$10,$11)
			RETURNING id`, in.Subject, in.Predicate, in.Object, text, vecLit(in.Embedding), conf, vf, exp, now, status,
			in.SourceEpisode).Scan(&id)
		return id, err
	}
	resp := map[string]any{"action": action}
	switch action {
	case ActCreated:
		id, e := insert("active")
		err = e
		resp["fact_id"] = fmt.Sprint(id)
	case ActHistory:
		id, e := insert("history")
		err = e
		resp["fact_id"] = fmt.Sprint(id)
	case ActReinforced:
		_, err = tx.Exec(ctx, `UPDATE facts SET confidence = LEAST(1.0, confidence + 0.05), last_confirmed=$2,
			expires_at = CASE WHEN $3::double precision IS NULL THEN expires_at
			                  ELSE GREATEST(COALESCE(expires_at, 0), $3) END WHERE id=$1`, curID, now, exp)
		resp["fact_id"] = fmt.Sprint(curID)
	case ActSuperseded:
		if _, err = tx.Exec(ctx, `UPDATE facts SET status='superseded' WHERE id=$1`, curID); err == nil {
			var id int64
			id, err = insert("active")
			resp["fact_id"], resp["replaced"] = fmt.Sprint(id), fmt.Sprint(curID)
			if err == nil {
				_, err = tx.Exec(ctx, `UPDATE facts SET superseded_by=$2 WHERE id=$1`, curID, id)
			}
		}
	case ActRejected:
		resp["fact_id"] = fmt.Sprint(curID)
	}
	if err != nil {
		fail(w, 500, err)
		return
	}
	if err = tx.Commit(ctx); err != nil {
		fail(w, 500, err)
		return
	}
	writeJSON(w, 200, resp)
}

func (s *Server) searchFacts(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Query     string    `json:"query"`
		Embedding []float64 `json:"embedding"`
		K         int       `json:"k"`
		Now       *float64  `json:"now"`
		Mode      string    `json:"mode"`
	}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		fail(w, 400, err)
		return
	}
	if in.K <= 0 {
		in.K = 5
	}
	out, err := s.hybrid(r.Context(), "facts", `status='active' AND (expires_at IS NULL OR expires_at > $1)`,
		[]any{nowOr(in.Now)}, in.Query, in.Embedding, in.K, in.Mode)
	if err != nil {
		fail(w, 500, err)
		return
	}
	writeJSON(w, 200, out)
}

// ---------------------------------------------------------------- procedural

func (s *Server) putSkill(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Name          string          `json:"name"`
		Source        string          `json:"source"`
		Docstring     string          `json:"docstring"`
		Params        json.RawMessage `json:"params"`
		Embedding     []float64       `json:"embedding"`
		OriginEpisode *string         `json:"origin_episode"`
		Gate          json.RawMessage `json:"gate"`
		Now           *float64        `json:"now"`
	}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		fail(w, 400, err)
		return
	}
	if in.Name == "" || in.Source == "" {
		fail(w, 400, fmt.Errorf("name and source are required"))
		return
	}
	now := nowOr(in.Now)
	ctx := r.Context()
	tx, err := s.db.Begin(ctx)
	if err != nil {
		fail(w, 500, err)
		return
	}
	defer tx.Rollback(ctx)
	if _, err = tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtext($1))`, in.Name); err != nil {
		fail(w, 500, err)
		return
	}
	var version int
	if err = tx.QueryRow(ctx, `SELECT COALESCE(MAX(version),0)+1 FROM skills WHERE name=$1`, in.Name).Scan(&version); err != nil {
		fail(w, 500, err)
		return
	}
	rows, err := tx.Query(ctx, `UPDATE skills SET status='superseded' WHERE name=$1 AND status='active' RETURNING version`, in.Name)
	if err != nil {
		fail(w, 500, err)
		return
	}
	var superseded []int
	for rows.Next() {
		var v int
		_ = rows.Scan(&v)
		superseded = append(superseded, v)
	}
	rows.Close()
	for _, v := range superseded {
		if err = s.event(ctx, tx, now, "superseded", in.Name, v, nil); err != nil {
			fail(w, 500, err)
			return
		}
	}
	params, gate := jsonOrNull(in.Params), jsonOrNull(in.Gate)
	if params == nil {
		params = "[]"
	}
	if gate == nil {
		gate = "{}"
	}
	if _, err = tx.Exec(ctx, `INSERT INTO skills(name,version,source,docstring,params,embedding,origin_episode,gate,status,created_at)
		VALUES($1,$2,$3,$4,$5::jsonb,$6::vector,$7,$8::jsonb,'active',$9)`,
		in.Name, version, in.Source, in.Docstring, params, vecLit(in.Embedding), in.OriginEpisode, gate, now); err != nil {
		fail(w, 500, err)
		return
	}
	if err = s.event(ctx, tx, now, "created", in.Name, version, nil); err != nil {
		fail(w, 500, err)
		return
	}
	if err = tx.Commit(ctx); err != nil {
		fail(w, 500, err)
		return
	}
	writeJSON(w, 201, map[string]any{"name": in.Name, "version": version, "status": "active"})
}

func (s *Server) searchSkills(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Query     string    `json:"query"`
		Embedding []float64 `json:"embedding"`
		K         int       `json:"k"`
	}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		fail(w, 400, err)
		return
	}
	if in.K <= 0 {
		in.K = 5
	}
	out, err := s.hybrid(r.Context(), "skills", `status='active'`, nil, in.Query, in.Embedding, in.K, "")
	if err != nil {
		fail(w, 500, err)
		return
	}
	writeJSON(w, 200, out)
}

func (s *Server) queryJSON(w http.ResponseWriter, r *http.Request, sql string, args ...any) {
	rows, err := s.db.Query(r.Context(), sql, args...)
	if err != nil {
		fail(w, 500, err)
		return
	}
	defer rows.Close()
	out := []json.RawMessage{}
	for rows.Next() {
		var raw []byte
		if err := rows.Scan(&raw); err != nil {
			fail(w, 500, err)
			return
		}
		out = append(out, raw)
	}
	writeJSON(w, 200, out)
}

func (s *Server) listSkills(w http.ResponseWriter, r *http.Request) {
	s.queryJSON(w, r, `SELECT to_jsonb(t) - 'embedding' - 'tsv' FROM skills t
		WHERE ($1 = '' OR status = $1) ORDER BY name, version`, r.URL.Query().Get("status"))
}

func (s *Server) listEvents(w http.ResponseWriter, r *http.Request) {
	s.queryJSON(w, r, `SELECT to_jsonb(e) - 'id' FROM skill_events e ORDER BY id`)
}

func (s *Server) recordUsage(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Version int      `json:"version"`
		Success bool     `json:"success"`
		Now     *float64 `json:"now"`
	}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		fail(w, 400, err)
		return
	}
	name, now, ctx := r.PathValue("name"), 0.0, r.Context()
	now = nowOr(in.Now)
	tx, err := s.db.Begin(ctx)
	if err != nil {
		fail(w, 500, err)
		return
	}
	defer tx.Rollback(ctx)
	var uses, succ int
	var status string
	err = tx.QueryRow(ctx, `UPDATE skills SET uses=uses+1, successes=successes+$3, last_used=$4
		WHERE name=$1 AND version=$2 RETURNING uses, successes, status`, name, in.Version, boolToInt(in.Success), now).
		Scan(&uses, &succ, &status)
	if err == pgx.ErrNoRows {
		writeJSON(w, 200, map[string]string{"status": "unknown"})
		return
	}
	if err != nil {
		fail(w, 500, err)
		return
	}
	ok := in.Success
	if err = s.event(ctx, tx, now, "used", name, in.Version, &ok); err != nil {
		fail(w, 500, err)
		return
	}
	if status == "active" && shouldDeprecate(uses, succ) {
		status = "deprecated"
		if _, err = tx.Exec(ctx, `UPDATE skills SET status='deprecated' WHERE name=$1 AND version=$2`, name, in.Version); err == nil {
			err = s.event(ctx, tx, now, "deprecated", name, in.Version, nil)
		}
		if err != nil {
			fail(w, 500, err)
			return
		}
	}
	if err = tx.Commit(ctx); err != nil {
		fail(w, 500, err)
		return
	}
	writeJSON(w, 200, map[string]any{"status": status, "uses": uses, "successes": succ})
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// ---------------------------------------------------------------- admin

func (s *Server) reap(ctx context.Context, now float64, idle *float64) (map[string]int64, error) {
	tag, err := s.db.Exec(ctx, `UPDATE facts SET status='expired' WHERE status='active' AND expires_at IS NOT NULL AND expires_at <= $1`, now)
	if err != nil {
		return nil, err
	}
	res := map[string]int64{"facts_expired": tag.RowsAffected(), "skills_retired": 0}
	if idle != nil && *idle > 0 {
		rows, err := s.db.Query(ctx, `UPDATE skills SET status='retired' WHERE status='active'
			AND $1 - COALESCE(last_used, created_at) > $2 RETURNING name, version`, now, *idle)
		if err != nil {
			return nil, err
		}
		type nv struct {
			n string
			v int
		}
		var done []nv
		for rows.Next() {
			var x nv
			_ = rows.Scan(&x.n, &x.v)
			done = append(done, x)
		}
		rows.Close()
		for _, x := range done {
			_, _ = s.db.Exec(ctx, `INSERT INTO skill_events(ts,type,name,version) VALUES($1,'retired',$2,$3)`, now, x.n, x.v)
		}
		res["skills_retired"] = int64(len(done))
	}
	return res, nil
}

func (s *Server) reapHandler(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Now          *float64 `json:"now"`
		SkillIdleTTL *float64 `json:"skill_idle_ttl"`
	}
	_ = json.NewDecoder(r.Body).Decode(&in)
	res, err := s.reap(r.Context(), nowOr(in.Now), in.SkillIdleTTL)
	if err != nil {
		fail(w, 500, err)
		return
	}
	writeJSON(w, 200, res)
}

func (s *Server) stats(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	out := map[string]any{}
	var eps int
	_ = s.db.QueryRow(ctx, `SELECT count(*) FROM episodes`).Scan(&eps)
	out["episodes"] = eps
	for _, t := range []string{"facts", "skills"} {
		m := map[string]int{}
		rows, err := s.db.Query(ctx, fmt.Sprintf(`SELECT status, count(*) FROM %s GROUP BY status`, t))
		if err != nil {
			fail(w, 500, err)
			return
		}
		for rows.Next() {
			var st string
			var n int
			_ = rows.Scan(&st, &n)
			m[st] = n
		}
		rows.Close()
		out[t] = m
	}
	writeJSON(w, 200, out)
}
