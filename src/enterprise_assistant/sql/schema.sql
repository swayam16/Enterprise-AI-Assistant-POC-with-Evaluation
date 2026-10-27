-- Knowledge base + support desk + CRM, all in one Postgres with pgvector.
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks CASCADE;
CREATE TABLE chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    url        TEXT NOT NULL,
    title      TEXT NOT NULL,
    section    TEXT NOT NULL,
    text       TEXT NOT NULL,
    -- Title and heading are weighted above body text for keyword search.
    tsv        tsvector GENERATED ALWAYS AS (
                   setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                   setweight(to_tsvector('english', coalesce(section, '')), 'A') ||
                   setweight(to_tsvector('english', coalesce(text, '')), 'B')
               ) STORED,
    embedding  vector(%(dim)s) NOT NULL
);

-- Seeded, fictional support desk and CRM that the MCP server exposes as tools.
DROP TABLE IF EXISTS tickets CASCADE;
DROP TABLE IF EXISTS contacts CASCADE;
DROP TABLE IF EXISTS accounts CASCADE;

CREATE TABLE accounts (
    account_id    TEXT PRIMARY KEY,
    name          TEXT UNIQUE NOT NULL,
    industry      TEXT NOT NULL,
    plan          TEXT NOT NULL CHECK (plan IN ('Free', 'Premium', 'Ultimate')),
    seats         INTEGER NOT NULL,
    arr_usd       INTEGER NOT NULL,
    region        TEXT NOT NULL,
    csm           TEXT NOT NULL,
    renewal_date  DATE NOT NULL,
    health_score  INTEGER NOT NULL CHECK (health_score BETWEEN 0 AND 100)
);

CREATE TABLE contacts (
    contact_id  TEXT PRIMARY KEY,
    account_id  TEXT NOT NULL REFERENCES accounts(account_id),
    name        TEXT NOT NULL,
    role        TEXT NOT NULL,
    email       TEXT NOT NULL,
    phone       TEXT NOT NULL
);

CREATE TABLE tickets (
    ticket_id     TEXT PRIMARY KEY,
    account_id    TEXT NOT NULL REFERENCES accounts(account_id),
    contact_id    TEXT NOT NULL REFERENCES contacts(contact_id),
    subject       TEXT NOT NULL,
    description   TEXT NOT NULL,
    product_area  TEXT NOT NULL,
    priority      TEXT NOT NULL CHECK (priority IN ('P1', 'P2', 'P3', 'P4')),
    status        TEXT NOT NULL CHECK (status IN ('open', 'pending', 'solved')),
    assignee      TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL,
    related_doc   TEXT  -- docs page the support engineer linked, if any
);
CREATE INDEX tickets_account_idx ON tickets (account_id, status);
