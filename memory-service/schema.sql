CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodes (
    id          text PRIMARY KEY,
    task_id     text,
    kind        text,
    success     boolean NOT NULL DEFAULT false,
    prompt      text,
    answer      text,
    tokens_in   integer,
    tokens_out  integer,
    cost        double precision,
    trajectory  jsonb,
    meta        jsonb,
    embedding   vector(256),
    created_at  double precision
);
CREATE INDEX IF NOT EXISTS episodes_vec ON episodes USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS facts (
    id              bigserial PRIMARY KEY,
    subject         text NOT NULL,
    predicate       text NOT NULL,
    object          text NOT NULL,
    text            text NOT NULL,
    embedding       vector(256),
    confidence      double precision NOT NULL DEFAULT 0.8,
    valid_from      double precision NOT NULL,
    expires_at      double precision,
    last_confirmed  double precision,
    status          text NOT NULL DEFAULT 'active',
    superseded_by   bigint,
    source_episode  text,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE UNIQUE INDEX IF NOT EXISTS facts_one_active ON facts (subject, predicate) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS facts_tsv ON facts USING gin (tsv);
CREATE INDEX IF NOT EXISTS facts_vec ON facts USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS skills (
    id              bigserial PRIMARY KEY,
    name            text NOT NULL,
    version         integer NOT NULL,
    source          text NOT NULL,
    docstring       text NOT NULL,
    params          jsonb NOT NULL DEFAULT '[]',
    embedding       vector(256),
    origin_episode  text,
    gate            jsonb NOT NULL DEFAULT '{}',
    status          text NOT NULL DEFAULT 'active',
    uses            integer NOT NULL DEFAULT 0,
    successes       integer NOT NULL DEFAULT 0,
    created_at      double precision NOT NULL,
    last_used       double precision,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', name || ' ' || docstring)) STORED,
    UNIQUE (name, version)
);
CREATE INDEX IF NOT EXISTS skills_tsv ON skills USING gin (tsv);

CREATE TABLE IF NOT EXISTS skill_events (
    id       bigserial PRIMARY KEY,
    ts       double precision NOT NULL,
    type     text NOT NULL,
    name     text NOT NULL,
    version  integer NOT NULL,
    success  boolean
);
