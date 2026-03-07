PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS cases (
  id_case               INTEGER PRIMARY KEY AUTOINCREMENT,
  image_name            TEXT NOT NULL,
  image_path            TEXT NOT NULL,
  original_description  TEXT,
  created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS results_patterns (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  id_case       INTEGER NOT NULL,
  model_name    TEXT NOT NULL DEFAULT 'patterns',
  model_version TEXT NOT NULL,
  pred1_label   TEXT NOT NULL,
  pred1_prob    REAL NOT NULL CHECK (pred1_prob >= 0 AND pred1_prob <= 1),
  pred2_label   TEXT,
  pred2_prob    REAL CHECK (pred2_prob IS NULL OR (pred2_prob >= 0 AND pred2_prob <= 1)),
  probs_json    TEXT,
  gradcam_json  TEXT,
  report_json   TEXT,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (id_case) REFERENCES cases(id_case) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS results_diseases (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  id_case       INTEGER NOT NULL,
  model_name    TEXT NOT NULL DEFAULT 'diseases',
  model_version TEXT NOT NULL,
  pred1_label   TEXT NOT NULL,
  pred1_prob    REAL NOT NULL CHECK (pred1_prob >= 0 AND pred1_prob <= 1),
  pred2_label   TEXT,
  pred2_prob    REAL CHECK (pred2_prob IS NULL OR (pred2_prob >= 0 AND pred2_prob <= 1)),
  probs_json    TEXT,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (id_case) REFERENCES cases(id_case) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cases_created_at
  ON cases(created_at);

CREATE INDEX IF NOT EXISTS idx_results_patterns_case
  ON results_patterns(id_case);

CREATE INDEX IF NOT EXISTS idx_results_diseases_case
  ON results_diseases(id_case);