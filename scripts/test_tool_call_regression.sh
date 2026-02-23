#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${MLX_CLUSTER_URL:-http://127.0.0.1:8080}"
API_KEY="${MLX_API_KEY:-none}"
MODEL="${MODEL_ID:-}"
OUT_DIR="${OUT_DIR:-/tmp/tool_call_regression}"
mkdir -p "$OUT_DIR"

if [[ -z "$MODEL" ]]; then
  MODEL="$(curl -s "${BASE_URL}/v1/models" | jq -r '.data[0].id // empty' || true)"
fi
MODEL="${MODEL:-GLM-4.7-Flash-8bit}"

pass=0
fail=0

report() {
  local name="$1" ok="$2" note="$3"
  if [[ "$ok" == "1" ]]; then
    pass=$((pass + 1))
    echo "[PASS] $name - $note"
  else
    fail=$((fail + 1))
    echo "[FAIL] $name - $note"
  fi
}

post_json() {
  local name="$1" body="$2"
  local header_file="$OUT_DIR/${name}.headers.txt"
  local body_file="$OUT_DIR/${name}.body.json"
  curl -sS --http1.1 --connect-timeout 5 --max-time 120 \
    -D "$header_file" -o "$body_file" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    "${BASE_URL}/v1/chat/completions" \
    -d "$body"
  echo "$header_file|$body_file"
}

echo "BASE_URL=$BASE_URL"
echo "MODEL=$MODEL"
echo "OUT_DIR=$OUT_DIR"

# 1) basic auto tool call
r=$(post_json "t1_basic_auto" "{
  \"model\":\"$MODEL\",
  \"stream\":false,
  \"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo? Use get_time timezone=Asia/Tokyo\"}],
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"Get time\",\"parameters\":{\"type\":\"object\",\"properties\":{\"timezone\":{\"type\":\"string\"}},\"required\":[\"timezone\"]}}}],
  \"tool_choice\":\"auto\"
}")
b="${r#*|}"
fr="$(jq -r '.choices[0].finish_reason // ""' "$b" 2>/dev/null || true)"
tc="$(jq -r '.choices[0].message.tool_calls[0].function.name // ""' "$b" 2>/dev/null || true)"
report "t1_basic_auto" "$([[ "$fr" == "tool_calls" && "$tc" == "get_time" ]] && echo 1 || echo 0)" "finish_reason=$fr tool=$tc"

# 2) forced tool call
r=$(post_json "t2_forced" "{
  \"model\":\"$MODEL\",
  \"stream\":false,
  \"messages\":[{\"role\":\"user\",\"content\":\"Give me any answer\"}],
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],
  \"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}}
}")
b="${r#*|}"
tc="$(jq -r '.choices[0].message.tool_calls[0].function.name // ""' "$b" 2>/dev/null || true)"
report "t2_forced" "$([[ "$tc" == "get_weather" ]] && echo 1 || echo 0)" "tool=$tc"

# 3) tool_choice none
r=$(post_json "t3_none" "{
  \"model\":\"$MODEL\",
  \"stream\":false,
  \"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo?\"}],
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"Get time\",\"parameters\":{\"type\":\"object\",\"properties\":{\"timezone\":{\"type\":\"string\"}},\"required\":[\"timezone\"]}}}],
  \"tool_choice\":\"none\"
}")
b="${r#*|}"
fr="$(jq -r '.choices[0].finish_reason // ""' "$b" 2>/dev/null || true)"
has_tc="$(jq -r 'if .choices[0].message.tool_calls then "yes" else "no" end' "$b" 2>/dev/null || true)"
report "t3_none" "$([[ "$fr" == "stop" && "$has_tc" == "no" ]] && echo 1 || echo 0)" "finish_reason=$fr has_tool_calls=$has_tc"

# 4) arg extraction variants
r=$(post_json "t4_args_cn" "{
  \"model\":\"$MODEL\",
  \"stream\":false,
  \"messages\":[{\"role\":\"user\",\"content\":\"帮我查北京天气，请调用 get_weather\"}],
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],
  \"tool_choice\":\"auto\"
}")
b="${r#*|}"
args="$(jq -r '.choices[0].message.tool_calls[0].function.arguments // ""' "$b" 2>/dev/null || true)"
city="$(echo "$args" | jq -r '.city // .location // empty' 2>/dev/null || true)"
report "t4_args_cn" "$([[ -n "$city" ]] && echo 1 || echo 0)" "arguments=$args"

r=$(post_json "t4_args_en" "{
  \"model\":\"$MODEL\",
  \"stream\":false,
  \"messages\":[{\"role\":\"user\",\"content\":\"Please call get_weather city=Berlin\"}],
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],
  \"tool_choice\":\"auto\"
}")
b="${r#*|}"
args="$(jq -r '.choices[0].message.tool_calls[0].function.arguments // ""' "$b" 2>/dev/null || true)"
city="$(echo "$args" | jq -r '.city // .location // empty' 2>/dev/null || true)"
report "t4_args_en" "$([[ -n "$city" ]] && echo 1 || echo 0)" "arguments=$args"

# 5) stream with tools
stream_file="$OUT_DIR/t5_stream.txt"
curl -sS -N --http1.1 --connect-timeout 5 --max-time 120 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  "${BASE_URL}/v1/chat/completions" \
  -d "{
    \"model\":\"$MODEL\",
    \"stream\":true,
    \"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo? Use get_time timezone=Asia/Tokyo\"}],
    \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"Get time\",\"parameters\":{\"type\":\"object\",\"properties\":{\"timezone\":{\"type\":\"string\"}},\"required\":[\"timezone\"]}}}],
    \"tool_choice\":\"auto\"
  }" > "$stream_file" || true

if rg -q 'tool_calls' "$stream_file" && rg -q '\[DONE\]' "$stream_file" && ! rg -q '<\|user\|>|<think>|</think>' "$stream_file"; then
  report "t5_stream" "1" "tool_calls + DONE + no leaked control tokens"
else
  report "t5_stream" "0" "check $stream_file"
fi

# 6) tool loop
first_body="$OUT_DIR/t6_loop_step1.json"
cp "$OUT_DIR/t1_basic_auto.body.json" "$first_body" 2>/dev/null || true
tool_id="$(jq -r '.choices[0].message.tool_calls[0].id // empty' "$first_body" 2>/dev/null || true)"
tool_name="$(jq -r '.choices[0].message.tool_calls[0].function.name // empty' "$first_body" 2>/dev/null || true)"
tool_args="$(jq -c '.choices[0].message.tool_calls[0].function.arguments // "{}"' "$first_body" 2>/dev/null || echo '"{}"')"

if [[ -n "$tool_id" && -n "$tool_name" ]]; then
  r=$(post_json "t6_loop_step2" "{
    \"model\":\"$MODEL\",
    \"stream\":false,
    \"messages\":[
      {\"role\":\"user\",\"content\":\"What time is it in Tokyo?\"},
      {\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"$tool_id\",\"type\":\"function\",\"function\":{\"name\":\"$tool_name\",\"arguments\":$tool_args}}]},
      {\"role\":\"tool\",\"tool_call_id\":\"$tool_id\",\"content\":\"{\\\"time\\\":\\\"15:30\\\",\\\"timezone\\\":\\\"Asia/Tokyo\\\"}\"}
    ],
    \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"Get time\",\"parameters\":{\"type\":\"object\",\"properties\":{\"timezone\":{\"type\":\"string\"}},\"required\":[\"timezone\"]}}}],
    \"tool_choice\":\"none\"
  }")
  b="${r#*|}"
  fr="$(jq -r '.choices[0].finish_reason // ""' "$b" 2>/dev/null || true)"
  txt="$(jq -r '.choices[0].message.content // ""' "$b" 2>/dev/null || true)"
  report "t6_tool_loop" "$([[ "$fr" == "stop" && -n "$txt" ]] && echo 1 || echo 0)" "finish_reason=$fr"
else
  report "t6_tool_loop" "0" "missing tool_call from step1"
fi

echo
echo "Summary: pass=$pass fail=$fail"
if [[ "$fail" -gt 0 ]]; then
  exit 1
fi
