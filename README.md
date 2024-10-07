# NLIR: Natural Language Intermediate Representation for Mechanized Theorem Proving

## Install

```
pip install -e .
```

To run the agent, you also need to install [coq-lsp](https://github.com/ejgallego/coq-lsp) >= 0.2.0 . On Unix-like systems, this is best done using the [OPAM package manager](https://opam.ocaml.org/), then do `opam install coq-lsp`.

## Getting started

The default configuration can be found in `conf/config.yaml`
To try the agent, first launch `pet-server` in a terminal

```bash
$ pet-server
```

Then (in another terminal)
```
$ python nlir-cli.py +file=foo.v +thm=foo
```

You should see each iteration of the proof in stdout.

Note: The default Coq workspace is `examples` which contains `foo.v` (see default config in `conf/config.yaml`).
You can change all these options with the `field=...` syntax (e.g., `+workspace=./examples`).

To replay a proof from a conversation log:

```
$ python nlir-cli.py +file=foo.v +thm=foo +replay=foo.v:foo_241007-174135.jsonl
```

## Benchmark

To try a complete benchmark (e.g., the one in `conf/benchmark/example.yaml`)

```
$ python nlir-bench.py
```

The conversation logs should be stored in `./outputs`

To launch one of the actual benchmark.
You need to clone it first (e.g., in `../lf` for logical_foundations), and then:

```
python nlir-bench.py workspace=../lf benchmark=logical_foundations
```

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
