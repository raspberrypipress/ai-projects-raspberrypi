#!/usr/bin/bash

curl --silent http://localhost:8000/api/chat \
     -H 'Content-Type: application/json' \
     -d @- <<EOF
{"model": "qwen2:1.5b",
 "messages": [
   {"role": "user",
    "content": "Translate to French: The cat is on the table."
   }]}
EOF

