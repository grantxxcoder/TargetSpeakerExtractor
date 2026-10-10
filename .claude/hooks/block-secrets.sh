#!/usr/bin/env bash
# PreToolUse guard: refuse any tool call that names a secrets file
# (.env, .env.*, .envrc, secrets.toml). Exit 2 blocks the call and the
# message on stderr goes back to Claude. Added 2026-10-09 after Claude read
# .env with sed, which the permissions.deny list (Read + cat only) let through.
#
# Checked: the Bash command text, and the path fields of the file tools.
# Grep's `pattern` is the text being searched FOR, not a file, so it is not
# checked. Not caught: a command that reads .env without naming it (e.g. a
# recursive grep over the repo root) -- only an OS sandbox closes that.
input=$(cat)
tool=$(jq -r '.tool_name // ""' <<<"$input")
if [ "$tool" = "Bash" ]; then
  text=$(jq -r '.tool_input.command // ""' <<<"$input")
elif [ "$tool" = "Glob" ]; then
  text=$(jq -r '[.tool_input.pattern, .tool_input.path] | map(select(. != null)) | join("\n")' <<<"$input")
else
  text=$(jq -r '[.tool_input.file_path, .tool_input.notebook_path, .tool_input.path, .tool_input.glob] | map(select(. != null)) | join("\n")' <<<"$input")
fi
# `.env` as a whole name (so os.environ, dotenv and tse_venv pass), any
# `.env.*` variant, `.envrc`, and `secrets.toml`.
if grep -qE '(^|[^[:alnum:]_])\.env($|[^[:alnum:]_])|\.envrc|secrets\.toml' <<<"$text"; then
  echo "BLOCKED by .claude/hooks/block-secrets.sh: this touches a secrets file (.env, .env.*, .envrc, secrets.toml). Never read, list, grep, source or edit it -- the user sets keys themselves." >&2
  exit 2
fi
exit 0
