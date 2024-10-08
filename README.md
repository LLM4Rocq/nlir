# NLIR: Natural Language Intermediate Representation for Mechanized Theorem Proving

## Install

To run the agent, you need to install [coq-lsp](https://github.com/ejgallego/coq-lsp) >= 0.2.0 . On Unix-like systems, this is best done using the [OPAM package manager](https://opam.ocaml.org/), then do `opam install coq-lsp`.

Then you can install NLIR with a simple:

```
pip install -e .
```

## Getting started

The default configuration can be found in `conf/config.yaml`
To try the agent, first launch `pet-server` in a terminal

```bash
$ pet-server
```

Then (in another terminal)
```
$ python nlir-cli.py file=foo.v theorem=foo
```

You should see each iteration of the proof in stdout.

We use [hydra](https://hydra.cc/docs/intro/) to manage the configurations.

```bash
$ python nlir-cli.py --help           
nlri-cli is powered by Hydra.

There are two possible modes:
- Use options `file=my_file.v` and `theorem=my_thm` to prove one theorem.
- Use option `benchmark=my_bench.yaml` to test a full benchmark.

Alternatively you can use your own config file with the option `--config-name myconf.yaml`.
Config files should be in the `conf` directory.

== Config ==
Override anything in the config (foo.bar=value)

workspace: examples
file: null
theorem: null
replay: false
benchmark: null
num_theorems: null
petanque:
  address: 127.0.0.1
  port: 8765
  timeout: 10
  run_opts: null
  context: false
agent:
  model_id: gpt-4o
  temperature: 1.0
search:
  kind: template
  mode: naive
  max_steps: 10


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```

## Examples

To replay the proof of `foo.v:foo` from a conversation log:

```
$ python nlir-cli.py file=foo.v theorem=foo +replay=foo_logs.jsonl
```

To try a theorem in a different workspace:

```
$ python nlir-cli.py workspace=../lf file=Induction.v theorem=add_comm
```

## Benchmark

Benchmarks are defined in `conf/benchmark` with the following format:

```yaml
- file: file1.v
  theorems:
    - thm11
    - thm12
- file: file2.v
  theorems:
    - thm21
    - thm22
    - thm23
```

To launch NLIR on a benchmark (e.g., `conf/benchmark/example.yaml`)

```
$ python nlir-cli.py benchmark=example
```

The conversation logs and the configuration will be stored in `./outputs/date/time/`