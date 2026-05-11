#!/usr/bin/env bash
set -euo pipefail

# Alapértelmezett bemenetek; felülírhatók környezeti változóval.
ORIGINAL_CONN_LOG_BACKUP="${ORIGINAL_CONN_LOG_BACKUP:-data/lab/zeek/conn.original_real_lab_001.log}"
RERUN_CONN_LOG="${RERUN_CONN_LOG:-data/lab/rerun_missing_flow/conn.log}"
OUTPUT_CONN_LOG="${OUTPUT_CONN_LOG:-data/lab/zeek/conn.log}"

for path in "$ORIGINAL_CONN_LOG_BACKUP" "$RERUN_CONN_LOG"; do
  if [[ ! -s "$path" ]]; then
    echo "[ERROR] Hiányzó vagy üres Zeek conn.log bemenet: $path" >&2
    exit 1
  fi
  if ! grep -q '^#fields' "$path"; then
    echo "[ERROR] A Zeek conn.log nem tartalmaz #fields sort: $path" >&2
    exit 1
  fi
done

if [[ "$OUTPUT_CONN_LOG" == "$ORIGINAL_CONN_LOG_BACKUP" || "$OUTPUT_CONN_LOG" == "$RERUN_CONN_LOG" ]]; then
  echo "[ERROR] Az output nem lehet azonos egyik bemenettel sem." >&2
  exit 1
fi

orig_rows="$(awk 'NF && $0 !~ /^#/ {count++} END {print count+0}' "$ORIGINAL_CONN_LOG_BACKUP")"
rerun_rows="$(awk 'NF && $0 !~ /^#/ {count++} END {print count+0}' "$RERUN_CONN_LOG")"
if [[ "$orig_rows" -eq 0 || "$rerun_rows" -eq 0 ]]; then
  echo "[ERROR] Mindkét conn.log fájlnak tartalmaznia kell valós adat sort. original=$orig_rows rerun=$rerun_rows" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CONN_LOG")"
tmp="$(mktemp "${OUTPUT_CONN_LOG}.tmp.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

# Eredeti Zeek fejléc az első adat sorig, #close nélkül.
awk '
  /^#close/ { next }
  /^#/ { print; next }
  { exit }
' "$ORIGINAL_CONN_LOG_BACKUP" > "$tmp"

# Valós conn sorok az eredeti és a rerun logból. Nem hozunk létre fake conn sort.
awk 'NF && $0 !~ /^#/ { print }' "$ORIGINAL_CONN_LOG_BACKUP" >> "$tmp"
awk 'NF && $0 !~ /^#/ { print }' "$RERUN_CONN_LOG" >> "$tmp"
printf '#close\t%s\n' "$(date -u +%Y-%m-%d-%H-%M-%S)" >> "$tmp"

mv "$tmp" "$OUTPUT_CONN_LOG"
trap - EXIT

echo "[OK] Eredeti conn sorok: $orig_rows"
echo "[OK] Rerun conn sorok: $rerun_rows"
echo "[OK] Kimenet: $OUTPUT_CONN_LOG"
