# NLIR: Natural Language Intermediate Representation for Mechanized Theorem Proving

NLIR leverage LLMs natural language reasoning ability for theorem proving with the Rocq interactive theorem prover (ITP).
We propose two interactive proof protocols both leveraging natural language reasoning: 

 - Tactic-by-tactic proof construction mimics the typical behavior of a standard Coq user: given the current goals, the agent generates one or several tactics that updates the goals and repeats this process until the proof is complete. 
 - Hierarchical proof templating tries to generate full proofs directly. Failed tactics are then replaced with holes to obtain a proof template. The agent repeats the process of filling each hole until the proof is complete. 

Our approach’s originality is that although both protocols’ inputs (goals) and outputs (tactics) are Coq code, the agent internally uses natural language as an intermediate representation to analyze the input and guide the code generation.
We couple both protocols with standard search algorithms leveraging feedback from the ITP and using natural language to rerank proof candidates.


## Install

To run the agent, you need to install [coq-lsp](https://github.com/ejgallego/coq-lsp) >= 0.2.0 . On Unix-like systems, this is best done using the [OPAM package manager](https://opam.ocaml.org/).
You can then install NLIR with pip.

```bash
$ opam install coq-lsp
$ pip install -e .
```

## Getting started

First launch `pet-server` in a terminal

```bash
$ pet-server
```

To communicate with the OpenAI API you need to export following environment variables (see https://platform.openai.com/ to generate the key):

```bash
export OPENAI_API_KEY="your secret key"
export OPENAI_PROJECT="your project id"
```

The default configuration can be found in `conf/config.yaml`.
You can override every field (see below).
E.g., to try the tactics agent without beam search on theorem `foo` defined in `examples/foo.v`:
```
$ python nlir-cli.py workspace=examples file=foo.v theorem=foo search.kind=tactics search.mode=naive
```

You should see each iteration of the proof in stdout.

## Configurations

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

workspace: examples # Path to the coq project
file: null          # file.v to prove a single theorem
theorem: null       # Theorem to prove a single theorem
replay: false       # log file to replay the conversation
benchmark: null     # benchmark suite
num_theorems: null  # stop after n theorem in the benchmark
log_dir: "logs"     # directory to store the log files
petanque:
  address: 127.0.0.1 # Address of the pet-server
  port: 8765         # port of the pet-server
  timeout: 10        # timeout for each tactic
  run_opts: null     # additional options for runtac
  context: false     # Add the beginning of the file without proofs in the prompt
agent:
  model_id: gpt-4o  # LLM id
  temperature: 1.0
search:
  kind: template # tactics or template
  mode: naive    # naive or beam
  max_steps: 10  # Number of steps for the search
  beam_size: 3   # only for beam search
  n_responses: 4 # only for beam search


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```

### Replay

To replay the proof of `foo.v:foo` from a conversation log using the default `config.yaml`.

```
$ python nlir-cli.py file=foo.v theorem=foo ++replay=logs/foo.v:foo_241105-093854.jsonl
```

### Benchmarks

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

To launch NLIR with the default configuration `config.yaml` on a benchmark (e.g., `conf/benchmark/example.yaml`)

```bash
$ python nlir-cli.py benchmark=example
```

The conversation logs and the configuration will be stored in `./outputs/date/time/`

### Using custom config files

You can also try one of the pre-defined benchmark configuration files (or write your own), and override some parameters, e.g.:

```bash
$ python nlir-cli.py --config-name conf_bb search.mode=naive
```

To run these benchmarks you need to download the [Logical Foundations](https://softwarefoundations.cis.upenn.edu/lf-current/index.html) and [Coq-BB5](https://github.com/ccz181078/Coq-BB5) in the parent directory (or change the `workspace` field in the config files).