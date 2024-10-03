# NLIR: Natural Language Intermediate Representation for Mechanized Theorem Proving

## Install

```
pip install -e .
```

To run the agent, you also need to install [coq-lsp](https://github.com/ejgallego/coq-lsp)

## Getting started

The configuration can be found in `conf/config.yaml`
To try the agent, first launch `pet-server` in a terminal

```bash
$ pet-server
```

Then (in another terminal)
```
$ python -m nlir
```

You should see each iteration of the proof in stdout.

The conversation logs should be stored in `./outputs`

## Config

Main config (in `conf/config.yaml`)

```bash
workspace: coq workspace root dir

petanque:
  address: pet-server address (default 127.0.0.1)
  port: pet-server port (default 8765)
  timeout: timeout for each tactic
  run_opts: runtime option for petanque

agent:
  kind: one of [gpt, ghost]
  model_id: OpenAI model-id (must be a chat model, e.g., gpt-4o)
  temperature: LLM temperature
  source_file: conversation log to replay for the ghost agent

search:
  kind: one of [tactics, template]
  mode: one of [naive, beam]
  max_steps: max iterations for the naive search

defaults:
  - benchmark: config file for the benchmark (see below)
```

Benchmark (in `conf/benchmark/`, see e.g., `example.yaml`)

```bash
- file: foo.v
  theorems:
    - foo
    - foofoo
- file: bar.v
  theorems:
    - bar
```

## Replayer

see `nlir/replay.py` for an example.