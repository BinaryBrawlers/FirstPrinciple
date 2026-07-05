"""Seed data for FirstPrinciple's Track A content graph.

Hand-authored HistoricalEpisode sequences for the two launch topics:
  - OS Memory Management  (six episodes)
  - Deep Learning         (nine episodes, task 4.2)

``seed_tracks_if_absent()`` is called from the FastAPI lifespan hook (task 4.3)
and writes each topic into Track A only when it is not already present.
"""

from datetime import date

import cognee  # type: ignore[import-untyped]

from memory.gateway import AgentRole, MemoryGateway
from models.schemas import HistoricalEpisode, Outcome, SourceConfidence

# ---------------------------------------------------------------------------
# OS Memory Management — six episodes in narrative order
# ---------------------------------------------------------------------------
# Dependency chain:
#   base_limit → segmentation → external_fragmentation (failure)
#                             → paging → page_tables → mmu_tlb

OS_MEMORY_EPISODES: list[HistoricalEpisode] = [
    HistoricalEpisode(
        id="os_mem_01_base_limit",
        concept="Base and Limit Registers",
        problem_posed=(
            "Early time-sharing systems loaded multiple programs into memory "
            "simultaneously.  How do you prevent one program from reading or "
            "overwriting another program's memory?"
        ),
        attempted_solution=(
            "Add two hardware registers to the CPU — a *base* register holding "
            "the starting physical address of the process's region and a *limit* "
            "register holding its length.  Every memory access is checked: "
            "if address < base or address >= base+limit, raise a protection fault."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "The hardware check is performed on every memory reference with no "
            "software overhead, giving strong isolation at minimal cost.  "
            "Processes cannot address memory outside their assigned range."
        ),
        requires=[],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Denning, P. J. (1970). Virtual Memory. ACM Computing Surveys.",
        published_date=date(1962, 1, 1),
    ),

    HistoricalEpisode(
        id="os_mem_02_segmentation",
        concept="Segmentation",
        problem_posed=(
            "Base-and-limit gives each process a single contiguous region.  "
            "Programs naturally decompose into logically distinct parts — code, "
            "stack, heap, shared libraries.  How do you protect and share these "
            "parts independently without merging them into one blob?"
        ),
        attempted_solution=(
            "Replace the single base-limit pair with a *segment table*: each "
            "entry holds a (base, limit, protection-bits) triple for a named "
            "logical segment.  A logical address is a (segment-number, offset) "
            "pair; the MMU looks up the segment table and translates it to a "
            "physical address, checking permissions en route."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Segmentation matches the programmer's mental model — code, data, "
            "and stack are separate — and enables fine-grained sharing and "
            "per-segment protection without requiring all segments to be "
            "contiguous in physical memory."
        ),
        requires=["os_mem_01_base_limit"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Burroughs B5000 Reference Manual, 1961; Multics design documents.",
        published_date=date(1961, 1, 1),
    ),

    HistoricalEpisode(
        id="os_mem_03_external_fragmentation",
        concept="External Fragmentation",
        problem_posed=(
            "Segmentation allocates variable-size chunks of physical memory for "
            "each segment.  After many allocations and frees, memory looks like "
            "Swiss cheese — lots of small free gaps between live segments.  "
            "A new large segment may not fit even though total free space is "
            "sufficient.  How do you handle this?"
        ),
        attempted_solution=(
            "Two approaches were tried: (1) *compaction* — periodically stop the "
            "world and slide all live segments together to coalesce free space; "
            "(2) *best-fit / first-fit allocators* — pick the gap that wastes the "
            "least space.  Neither eliminated the problem; they only delayed it."
        ),
        outcome=Outcome.FAILURE,
        why=(
            "Compaction is prohibitively expensive on large memories — copying "
            "gigabytes takes seconds during which processes are paused.  "
            "Allocator heuristics reduce fragmentation but cannot prevent it "
            "theoretically.  The root cause is that variable-size allocation "
            "necessarily creates unpredictably sized gaps."
        ),
        requires=["os_mem_02_segmentation"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Knuth, D. E. (1968). The Art of Computer Programming, Vol. 1, §2.5.",
        published_date=date(1968, 1, 1),
    ),

    HistoricalEpisode(
        id="os_mem_04_paging",
        concept="Paging",
        problem_posed=(
            "External fragmentation arises because segments are variable in size.  "
            "What if physical memory were divided into *fixed-size* chunks so that "
            "any free chunk can satisfy any request of the same size?"
        ),
        attempted_solution=(
            "Divide physical memory into fixed-size *frames* (e.g., 4 KiB) and "
            "divide each process's virtual address space into same-size *pages*.  "
            "A *page table* maps each virtual page number to a physical frame "
            "number.  The OS maintains a free-frame list; allocation is O(1) — "
            "pop any free frame and map the page to it."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Fixed-size units eliminate external fragmentation entirely: every "
            "free frame is interchangeable.  Internal fragmentation (wasted space "
            "within the last page of a segment) is bounded by page_size - 1 bytes "
            "per allocation, which is acceptable for typical page sizes."
        ),
        requires=["os_mem_03_external_fragmentation"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Atlas computer design, Manchester University, 1962; Fotheringham (1961).",
        published_date=date(1962, 1, 1),
    ),

    HistoricalEpisode(
        id="os_mem_05_page_tables",
        concept="Page Tables",
        problem_posed=(
            "A flat page table for a 32-bit address space with 4 KiB pages "
            "requires 2^20 entries × 4 bytes = 4 MiB *per process* just for the "
            "mapping metadata — and most of that space is unused sparse mappings.  "
            "How do you store page tables efficiently?"
        ),
        attempted_solution=(
            "Three complementary techniques emerged: (1) *multi-level page tables* "
            "— a tree of page-directory and page-table pages so that absent regions "
            "need no storage; (2) *inverted page tables* — one global table indexed "
            "by frame number rather than per-process virtual page; "
            "(3) *hashed page tables* — hash the virtual page number to reduce "
            "average lookup cost."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Multi-level tables are now universal (x86-64 uses four levels): they "
            "store only the mappings that actually exist, so a sparse address space "
            "costs proportionally little.  The trade-off is that a page fault or "
            "TLB miss now requires multiple memory accesses to walk the tree."
        ),
        requires=["os_mem_04_paging"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="x86 Architecture Programmer's Manual; Silberschatz OS textbook ch. 9.",
        published_date=date(1985, 1, 1),
    ),

    HistoricalEpisode(
        id="os_mem_06_mmu_tlb",
        concept="MMU and TLB",
        problem_posed=(
            "Every virtual memory access now requires one or more page-table "
            "lookups in RAM — a 4-level x86-64 walk costs four additional memory "
            "reads before the actual access.  This would make virtual memory "
            "five times slower than physical memory.  How do you recover "
            "near-native performance?"
        ),
        attempted_solution=(
            "Add a *Translation Lookaside Buffer (TLB)* — a small, fully "
            "associative cache inside the MMU that stores the most recently used "
            "virtual-to-physical page mappings.  On every address translation, "
            "the TLB is checked first; a *hit* (>99 % of the time for typical "
            "workloads) delivers the physical address in one cycle.  Only a "
            "*miss* falls through to the page-table walk."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Locality of reference means that a small TLB (64–1024 entries) "
            "achieves very high hit rates.  Modern CPUs have multi-level TLBs "
            "(L1 iTLB, L1 dTLB, L2 unified TLB) and hardware page-table walkers, "
            "making virtual-to-physical translation essentially free in the "
            "common case."
        ),
        requires=["os_mem_05_page_tables"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Clark & Emer (1985). Performance of the VAX-11/780 Translation Buffer.",
        published_date=date(1985, 1, 1),
    ),
]


# ---------------------------------------------------------------------------
# Deep Learning — nine episodes in narrative order
# ---------------------------------------------------------------------------
# Dependency chain:
#   perceptron → xor_failure → mlp → backpropagation
#             → cnn (concurrent: rnn) → vanishing_gradient (failure)
#             → lstm → attention → transformer

DEEP_LEARNING_EPISODES: list[HistoricalEpisode] = [
    HistoricalEpisode(
        id="dl_01_perceptron",
        concept="Perceptron",
        problem_posed=(
            "Can a machine learn to classify patterns by adjusting weights "
            "based on its own mistakes, rather than requiring explicit "
            "hand-coded rules?"
        ),
        attempted_solution=(
            "Rosenblatt's perceptron: a single layer of weighted inputs summed "
            "and passed through a threshold step function.  A learning rule "
            "increments weights on correct-class inputs and decrements them on "
            "misclassified inputs, iterating until convergence."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "The perceptron convergence theorem proves it finds a separating "
            "hyperplane for any linearly separable dataset in finite steps.  "
            "It demonstrated that gradient-free, error-driven weight updates "
            "could produce a learning machine from first principles."
        ),
        requires=[],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. Psychological Review.",
        published_date=date(1958, 1, 1),
    ),

    HistoricalEpisode(
        id="dl_02_xor_failure",
        concept="XOR Problem and Perceptron Limitations",
        problem_posed=(
            "The perceptron only classifies linearly separable problems.  "
            "XOR is the simplest function that is not linearly separable: no "
            "single straight line can separate its four truth-table rows into "
            "two classes.  Does this doom the entire neural-network program?"
        ),
        attempted_solution=(
            "Minsky and Papert proved formally that a single-layer perceptron "
            "cannot compute XOR.  Their analysis generalised to show that "
            "many interesting functions are beyond a one-layer threshold "
            "unit — casting doubt on the whole connectionist approach."
        ),
        outcome=Outcome.FAILURE,
        why=(
            "The proof was mathematically correct and the conclusion followed "
            "logically: single-layer linear classifiers cannot represent "
            "non-linear decision boundaries.  The result triggered the first "
            "AI winter for neural networks because no principled multi-layer "
            "learning algorithm existed yet."
        ),
        requires=["dl_01_perceptron"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="Minsky, M. & Papert, S. (1969). Perceptrons: An Introduction to Computational Geometry. MIT Press.",
        published_date=date(1969, 1, 1),
    ),

    HistoricalEpisode(
        id="dl_03_mlp",
        concept="Multi-Layer Perceptron (MLP)",
        problem_posed=(
            "If one layer cannot represent XOR, what happens with two or more "
            "layers of threshold units?  Can a hidden layer learn an internal "
            "representation that transforms the problem into a linearly "
            "separable one?"
        ),
        attempted_solution=(
            "Stack multiple layers of neurons, each applying a non-linear "
            "activation (initially sigmoid).  The universal approximation "
            "theorem later confirmed that a two-layer network with enough "
            "hidden units can approximate any continuous function on a compact "
            "domain — but no efficient weight-learning algorithm was known yet."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "The architecture itself was sound: hidden layers compose non-linear "
            "transformations so the network can carve out arbitrary decision "
            "boundaries.  The remaining obstacle was computing how to adjust the "
            "hidden-layer weights — credit assignment across layers."
        ),
        requires=["dl_02_xor_failure"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning Representations by Back-propagating Errors. Nature.",
        published_date=date(1986, 1, 1),
    ),

    HistoricalEpisode(
        id="dl_04_backpropagation",
        concept="Backpropagation",
        problem_posed=(
            "Given a multi-layer network with differentiable activations and a "
            "scalar loss, how do you efficiently compute the gradient of the "
            "loss with respect to every weight, including those in deep hidden "
            "layers far from the output?"
        ),
        attempted_solution=(
            "Apply the chain rule of calculus layer by layer, starting from "
            "the output and propagating error signals backward through the "
            "network.  Each layer receives the upstream gradient, multiplies "
            "by the local Jacobian (derivative of its activation), and passes "
            "the result further back — O(weights) total compute, the same "
            "order as a forward pass."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Backpropagation solved the credit-assignment problem exactly: "
            "every weight receives a precise gradient signal regardless of "
            "depth.  Combined with stochastic gradient descent it made deep "
            "networks trainable in principle, unlocking the MLP potential and "
            "setting the algorithmic foundation for all subsequent deep learning."
        ),
        requires=["dl_03_mlp"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning Representations by Back-propagating Errors. Nature.",
        published_date=date(1986, 10, 9),
    ),

    HistoricalEpisode(
        id="dl_05_cnn",
        concept="Convolutional Neural Network (CNN)",
        problem_posed=(
            "Fully connected networks applied to images treat each pixel as an "
            "independent feature, ignoring spatial locality and translation "
            "invariance.  A cat in the top-left corner looks completely "
            "different to the weight matrix than a cat in the bottom-right.  "
            "How do you build spatial awareness into the architecture?"
        ),
        attempted_solution=(
            "Share weights across spatial positions using learned convolutional "
            "filters: the same small kernel slides over every location, "
            "detecting the same local pattern regardless of where it appears.  "
            "Stack convolution → activation → pooling blocks to build "
            "progressively more abstract spatial features."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Weight sharing drastically reduces parameters while encoding the "
            "inductive bias that nearby pixels are related and that the same "
            "feature detector is useful everywhere.  LeCun's LeNet-5 achieved "
            "near-human digit recognition in 1998, decades before the GPU era "
            "made the idea scalable to ImageNet."
        ),
        requires=["dl_04_backpropagation"],
        concurrent_with=["dl_06_rnn"],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE.",
        published_date=date(1998, 1, 1),
    ),

    HistoricalEpisode(
        id="dl_06_rnn",
        concept="Recurrent Neural Network (RNN)",
        problem_posed=(
            "Feedforward networks process each input independently.  Language, "
            "speech, and time-series data have temporal dependencies: the "
            "meaning of a word depends on the words before it.  How do you "
            "give a network memory across time steps?"
        ),
        attempted_solution=(
            "Feed a hidden state vector from the previous time step back into "
            "the current computation alongside the new input.  The network "
            "learns to accumulate relevant history in this hidden state and "
            "to ignore irrelevant information — in theory.  Training uses "
            "backpropagation through time (BPTT), unrolling the RNN into a "
            "deep feedforward graph."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "RNNs could, in principle, capture arbitrary-length temporal "
            "dependencies with a fixed-size hidden state.  They were the "
            "dominant sequence model through the 1990s and early 2000s for "
            "speech recognition and language modelling, despite practical "
            "training difficulty on long sequences."
        ),
        requires=["dl_04_backpropagation"],
        concurrent_with=["dl_05_cnn"],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Elman, J. L. (1990). Finding Structure in Time. Cognitive Science.",
        published_date=date(1990, 1, 1),
    ),

    HistoricalEpisode(
        id="dl_07_vanishing_gradient",
        concept="Vanishing Gradient Problem",
        problem_posed=(
            "In deep networks and unrolled RNNs, backpropagation must multiply "
            "many Jacobian matrices together — one per layer or time step.  "
            "When these matrices have spectral radius < 1, the gradient "
            "shrinks exponentially toward zero as it travels backward.  "
            "Early layers (or early time steps) receive essentially no "
            "learning signal.  How do you train very deep or very long networks?"
        ),
        attempted_solution=(
            "Hochreiter's 1991 diploma thesis characterised the problem "
            "precisely: the gradient of the loss with respect to weights deep "
            "in the network (or far back in time) vanishes or explodes "
            "geometrically with depth.  Proposed fixes — careful weight "
            "initialisation, truncated BPTT, second-order methods — only "
            "partially alleviated the problem, leaving long-range dependency "
            "learning largely unsolved for plain RNNs."
        ),
        outcome=Outcome.FAILURE,
        why=(
            "The root cause is the chain rule applied to many near-zero "
            "derivatives: sigmoid saturates with derivative ≈ 0 away from "
            "zero; repeated multiplication drives the gradient to zero "
            "faster than any fixed-depth network could recover.  Plain RNNs "
            "were empirically confirmed unable to learn dependencies spanning "
            "more than ~10 time steps."
        ),
        requires=["dl_05_cnn", "dl_06_rnn"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis, TU Munich. / Bengio et al. (1994). Learning Long-Term Dependencies with Gradient Descent is Difficult. IEEE Transactions on Neural Networks.",
        published_date=date(1994, 1, 1),
    ),

    HistoricalEpisode(
        id="dl_08_lstm",
        concept="Long Short-Term Memory (LSTM)",
        problem_posed=(
            "Given the vanishing-gradient problem, how do you design a "
            "recurrent unit that can selectively remember information across "
            "hundreds of time steps without gradients vanishing?"
        ),
        attempted_solution=(
            "Introduce a *cell state* — a highway that carries information "
            "through time with additive (not multiplicative) updates, "
            "preventing gradient decay.  Three learned gates regulate "
            "information flow: a *forget gate* erases stale content, an "
            "*input gate* writes new content, and an *output gate* exposes "
            "a filtered view.  Gradients flow through the cell state with "
            "near-constant magnitude."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "The constant-error carousel (additive cell updates) solved the "
            "vanishing gradient problem for recurrent connections: the gradient "
            "of a loss at time T with respect to cell content at time T-k "
            "no longer decays exponentially with k.  LSTMs became the "
            "workhorse for speech, handwriting, and language models through "
            "the 2000s and early 2010s."
        ),
        requires=["dl_07_vanishing_gradient"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.",
        published_date=date(1997, 1, 1),
    ),

    HistoricalEpisode(
        id="dl_09_attention",
        concept="Attention Mechanism",
        problem_posed=(
            "Sequence-to-sequence models (e.g., neural machine translation) "
            "compress an entire variable-length input into a single fixed-size "
            "context vector before decoding.  For long sentences this bottleneck "
            "loses detail: the decoder cannot inspect earlier encoder states "
            "when generating a word that depends on a specific input position.  "
            "How do you let the decoder focus on the relevant part of the input "
            "at each output step?"
        ),
        attempted_solution=(
            "At each decoding step, compute a *soft alignment* (attention "
            "weights) over all encoder hidden states by scoring their "
            "compatibility with the current decoder state.  Take a weighted "
            "sum of encoder states as the *context vector* for this step — "
            "the model learns to attend to the most relevant positions.  "
            "Weights are computed by a small learned scoring network and "
            "normalised with softmax."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Attention lets the decoder directly access any encoder position "
            "with O(n) queries per step, bypassing the fixed-size bottleneck.  "
            "It also provides an interpretable alignment matrix.  The "
            "mechanism improved BLEU scores substantially and was immediately "
            "adopted across sequence tasks, eventually motivating the "
            "Transformer's full replacement of recurrence with attention."
        ),
        requires=["dl_08_lstm"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR.",
        published_date=date(2015, 1, 1),
    ),

    HistoricalEpisode(
        id="dl_10_transformer",
        concept="Transformer",
        problem_posed=(
            "RNNs and LSTMs process tokens sequentially: token t cannot be "
            "computed until token t-1 is done, preventing parallelism over "
            "sequence length.  Attention already captures long-range "
            "dependencies — why keep the sequential recurrence at all?  "
            "Can you build a sequence model using attention alone?"
        ),
        attempted_solution=(
            "Replace recurrence entirely with *multi-head self-attention*: "
            "every position attends to every other position in O(n²) "
            "operations, all computed in parallel.  Add position encodings "
            "to inject sequence order, feed-forward sublayers for "
            "position-wise transformation, residual connections, and layer "
            "normalisation.  Stack N encoder and N decoder blocks."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Full parallelism over sequence length enables training on orders "
            "of magnitude more data with the same wall-clock budget.  "
            "Self-attention computes direct dependencies between any two "
            "positions in a single layer, whereas RNNs need O(n) steps for "
            "the same span.  The Transformer became the backbone of every "
            "modern large language model — BERT, GPT, T5, and beyond."
        ),
        requires=["dl_09_attention"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.",
        published_date=date(2017, 1, 1),
    ),
]


# ---------------------------------------------------------------------------
# Lifespan hook — seed Track A on first startup (Requirements 10.3, 2.4, 2.5)
# ---------------------------------------------------------------------------

# Mapping of (query string, episode list) for each seed topic.
# The query string is used to probe Track A before writing.
_SEED_TOPICS: list[tuple[str, list[HistoricalEpisode]]] = [
    ("OS memory management", OS_MEMORY_EPISODES),
    ("deep learning", DEEP_LEARNING_EPISODES),
]


async def seed_tracks_if_absent() -> None:
    """Write hand-authored seed episodes into Track A if they are not yet present.

    Called once from the FastAPI lifespan startup hook.  For each seed topic,
    the function first queries Track A via ``cognee.recall()``; if the recall
    result is empty (or not found), it writes the episode list through
    ``MemoryGateway(AgentRole.INGESTION)`` with ``temporal_cognify=True`` and
    then runs ``cognee.consolidate_entity_descriptions_pipeline()`` to merge
    any duplicate entity descriptions.

    This function is idempotent: calling it multiple times will not create
    duplicate episodes because the recall check gates each write.

    Requirements: 10.3, 2.4, 2.5
    """
    gateway = MemoryGateway(AgentRole.INGESTION)

    for query, episodes in _SEED_TOPICS:
        # Check whether this topic already exists in Track A.
        already_present = False
        try:
            results = await cognee.recall(
                graph_name="content_track",
                query=query,
            )
            # Non-empty results indicate the topic has already been seeded.
            if results:
                already_present = True
        except Exception:
            # If recall fails (e.g. graph does not exist yet), treat as absent
            # and proceed with seeding.
            already_present = False

        if already_present:
            continue

        # Write episodes to Track A via the Ingestion role gateway.
        # Requirement 2.4: temporal_cognify=True preserves temporal ordering.
        await gateway.add_data_points(episodes, temporal_cognify=True)

        # Requirement 2.5: merge duplicate entity descriptions across sources.
        await cognee.consolidate_entity_descriptions_pipeline()
