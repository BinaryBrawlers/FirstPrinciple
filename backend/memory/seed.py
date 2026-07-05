"""
Seed episode sequences for FirstPrinciple.

OS_MEMORY_EPISODES covers the historical development of OS memory management
in narrative order, conforming to Requirement 10.1.

Do NOT add seed_tracks_if_absent() here — that belongs in task 4.3.
"""

from datetime import date

from models.schemas import HistoricalEpisode, Outcome, SourceConfidence

# ---------------------------------------------------------------------------
# OS Memory Management — six episodes in narrative order
# Requirement 10.1
# ---------------------------------------------------------------------------

OS_MEMORY_EPISODES: list[HistoricalEpisode] = [
    # 1. Base + Limit Registers
    HistoricalEpisode(
        id="os_mem_01_base_limit",
        concept="Base and limit registers",
        problem_posed=(
            "Early computers ran one program at a time with full access to physical memory. "
            "How can multiple programs share memory safely, each believing it owns a contiguous "
            "address space, without one program corrupting another?"
        ),
        attempted_solution=(
            "Hardware designers added two registers to the CPU: a base register holding the "
            "start address of a program's memory region and a limit register holding its size. "
            "Every memory access is checked: if (address < base) or (address >= base + limit), "
            "the hardware raises a protection fault."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "The scheme works for single-program time-sharing — each process gets a distinct "
            "base/limit pair loaded by the OS on context switch, providing isolation at near-zero "
            "hardware cost. It became the foundation for protected-mode memory in early batch OSes."
        ),
        requires=[],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Silberschatz, Galvin & Gagne — Operating System Concepts, 10th ed., Chapter 9",
        published_date=date(1961, 1, 1),
    ),

    # 2. Segmentation
    HistoricalEpisode(
        id="os_mem_02_segmentation",
        concept="Segmentation",
        problem_posed=(
            "Base/limit registers treat a program as one flat block, but real programs have "
            "logically distinct regions: code, stack, heap, and data. Sharing a library's code "
            "between processes means duplicating it. How can the OS map each logical region "
            "independently and allow sharing of code segments?"
        ),
        attempted_solution=(
            "Segmentation divides a process's address space into named, variable-length segments "
            "(code, data, stack, etc.), each described by a segment descriptor containing a base "
            "address and limit. A segment table maps segment numbers to descriptors. The CPU "
            "translates a logical address (segment number, offset) to a physical address by "
            "looking up the segment table and adding the offset to the base."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Segmentation enables fine-grained protection (each segment has its own read/write/"
            "execute bits) and efficient sharing (two processes can point their code segment "
            "descriptors at the same physical pages). It was implemented in Multics (1965) and "
            "later in x86 real and protected modes."
        ),
        requires=["os_mem_01_base_limit"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Corbató & Vyssotsky — Introduction and Overview of the Multics System (AFIPS 1965)",
        published_date=date(1965, 1, 1),
    ),

    # 3. External Fragmentation (failure)
    HistoricalEpisode(
        id="os_mem_03_external_fragmentation",
        concept="External fragmentation",
        problem_posed=(
            "Segmentation allocates variable-size chunks of physical memory. Over time, as "
            "segments are created and destroyed, free memory becomes scattered in small, "
            "non-contiguous holes. A new segment may need 1 MB of contiguous space, but only "
            "2 MB of free memory exists — split across dozens of tiny gaps. What happens?"
        ),
        attempted_solution=(
            "Compaction (moving all live segments together to coalesce free space) was proposed "
            "as a fix, but it requires stopping all processes, copying potentially gigabytes of "
            "data, and updating every segment descriptor — an enormous runtime cost. In practice "
            "systems either over-provisioned memory or crashed under heavy mixed-workload allocation."
        ),
        outcome=Outcome.FAILURE,
        why=(
            "Compaction is too expensive for interactive or real-time systems. External "
            "fragmentation is an intrinsic pathology of variable-size allocation: no allocation "
            "policy (first-fit, best-fit, worst-fit) eliminates it in the general case. The "
            "failure of segmentation to solve fragmentation was the direct engineering motivation "
            "for fixed-size page frames."
        ),
        requires=["os_mem_02_segmentation"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Knuth — The Art of Computer Programming, Vol. 1, §2.5 (Dynamic Storage Allocation)",
        published_date=date(1968, 1, 1),
    ),

    # 4. Paging
    HistoricalEpisode(
        id="os_mem_04_paging",
        concept="Paging",
        problem_posed=(
            "External fragmentation is caused by variable-size allocation. What if physical "
            "memory were divided into fixed-size frames and logical memory into equal-size pages, "
            "so any free frame can satisfy any page request regardless of where it sits in "
            "physical memory?"
        ),
        attempted_solution=(
            "Paging maps every fixed-size logical page (e.g., 4 KB) to any available physical "
            "frame of the same size. The OS maintains a page table per process translating page "
            "numbers to frame numbers. Because all units are the same size, any free frame fits "
            "any page — no contiguous physical allocation is ever needed."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Paging eliminates external fragmentation entirely. Internal fragmentation (wasted "
            "space in the last page) is at most one page per segment — bounded and acceptable. "
            "The scheme also simplifies physical memory management to a free-frame bitmap. Paging "
            "was first used in the Atlas computer (1962) and became universal with virtual memory."
        ),
        requires=["os_mem_03_external_fragmentation"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Kilburn et al. — One-Level Storage System, IRE Transactions on Electronic Computers (1962)",
        published_date=date(1962, 1, 1),
    ),

    # 5. Page Tables
    HistoricalEpisode(
        id="os_mem_05_page_tables",
        concept="Page tables",
        problem_posed=(
            "A flat page table for a 32-bit address space with 4 KB pages needs 2^20 entries "
            "(≈ 4 MB per process). With hundreds of processes, page tables themselves consume "
            "significant memory. For 64-bit address spaces the problem is catastrophic. How can "
            "page tables be made space-efficient while remaining fast to look up?"
        ),
        attempted_solution=(
            "Multiple strategies emerged: (1) Multi-level page tables (two- or four-level on "
            "x86-64) allocate only the table nodes that map actually-used address ranges, leaving "
            "huge sparse regions unallocated. (2) Inverted page tables store one entry per "
            "physical frame rather than per virtual page, capping size at physical RAM. "
            "(3) Hashed page tables provide O(1) lookup for sparse address spaces."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Multi-level page tables became the dominant design because they align naturally with "
            "sparse virtual address layouts (code, heap, stack are far apart) and integrate "
            "cleanly with hardware MMU walkers. The x86-64 four-level table (PML4 → PDP → PD → PT) "
            "is the direct descendant, handling 48-bit virtual addresses with only the needed "
            "intermediate nodes allocated."
        ),
        requires=["os_mem_04_paging"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Intel 64 and IA-32 Architectures Software Developer's Manual, Vol. 3A, §4.5",
        published_date=date(1985, 1, 1),
    ),

    # 6. MMU and TLB
    HistoricalEpisode(
        id="os_mem_06_mmu_tlb",
        concept="MMU and TLB",
        problem_posed=(
            "Every memory access with paging requires at least one extra memory read to walk the "
            "page table before reaching the actual data. A multi-level page table walk means 3–4 "
            "additional reads per access. How can virtual-to-physical translation be made fast "
            "enough not to double or quadruple memory latency?"
        ),
        attempted_solution=(
            "The Memory Management Unit (MMU) is a hardware block integrated into the CPU that "
            "performs address translation automatically. It caches recent translations in the "
            "Translation Lookaside Buffer (TLB) — a small, fully-associative cache of "
            "(virtual-page → physical-frame) entries. On a TLB hit the translation completes in "
            "one cycle; on a miss the MMU hardware-walks the page table and fills the TLB. "
            "Context switches flush or tag-stamp TLB entries with an Address Space ID (ASID)."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "TLB hit rates above 99% are typical for real workloads (working-set locality), "
            "making the amortised translation cost negligible. ASID tagging (introduced by MIPS "
            "R2000 in 1985) allows TLB entries to survive context switches, eliminating full "
            "flushes and dramatically cutting context-switch overhead. The MMU+TLB architecture "
            "is present in every modern CPU and is the reason paging is practical at scale."
        ),
        requires=["os_mem_05_page_tables"],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source=(
            "Patterson & Hennessy — Computer Organization and Design, 5th ed., §5.7; "
            "MIPS R2000 Architecture Reference Manual (Kane, 1987)"
        ),
        published_date=date(1987, 1, 1),
    ),
]


# ---------------------------------------------------------------------------
# Deep Learning — ten episodes in narrative order (nine narrative steps;
# CNN and RNN are concurrent and each get their own HistoricalEpisode object)
# Requirement 10.2
# ---------------------------------------------------------------------------

DEEP_LEARNING_EPISODES: list[HistoricalEpisode] = [
    # 1. Perceptron
    HistoricalEpisode(
        id="dl_01_perceptron",
        concept="Perceptron",
        problem_posed=(
            "In the mid-1950s, could a machine learn to classify patterns from examples, without "
            "being explicitly programmed with rules? Rosenblatt posed the question: can a simple "
            "weighted summing unit, adjusted after each mistake, converge to a correct decision "
            "boundary for linearly separable inputs?"
        ),
        attempted_solution=(
            "Frank Rosenblatt's Perceptron (1958) is a single artificial neuron: it computes a "
            "weighted sum of its binary inputs, applies a threshold activation, and outputs 0 or 1. "
            "The Perceptron Learning Rule adjusts each weight by adding the product of the learning "
            "rate, the error, and the corresponding input — a simple online update that is now "
            "recognised as stochastic gradient descent on a linear threshold unit."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Rosenblatt proved the Perceptron Convergence Theorem: if the data are linearly "
            "separable, the algorithm is guaranteed to find a solution in a finite number of steps. "
            "The Cornell Mark I Perceptron hardware (1960) demonstrated real-time image "
            "classification, generating enormous excitement about machine intelligence and "
            "establishing the foundational learning-rule paradigm that all later neural networks inherit."
        ),
        requires=[],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="Rosenblatt, F. — The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. Psychological Review, 65(6), 386–408 (1958)",
        published_date=date(1958, 1, 1),
    ),

    # 2. XOR Failure (FAILURE)
    HistoricalEpisode(
        id="dl_02_xor_failure",
        concept="XOR failure of single-layer perceptrons",
        problem_posed=(
            "If the Perceptron can learn any linearly separable function, what are its limits? "
            "Can a single-layer perceptron learn the XOR function — where class 1 is assigned to "
            "(0,1) and (1,0) but class 0 to (0,0) and (1,1)? More broadly, which Boolean "
            "functions are beyond its reach?"
        ),
        attempted_solution=(
            "Minsky and Papert systematically analysed the computational geometry of single-layer "
            "perceptrons. They showed that XOR requires a non-linear decision boundary (the two "
            "class-1 points cannot be separated from the two class-0 points by any hyperplane). "
            "Their 1969 book 'Perceptrons' extended this to many other functions and introduced "
            "formal order-of-predicate analysis, arguing that scaling to higher-order predicates "
            "was computationally intractable."
        ),
        outcome=Outcome.FAILURE,
        why=(
            "The proof that a single linear threshold unit cannot represent XOR exposed a "
            "fundamental architectural limitation. Minsky and Papert's pessimistic framing about "
            "multi-layer networks — combined with a perception that their critique applied broadly "
            "— contributed to the first AI winter, drastically cutting neural-network funding "
            "through the 1970s. The episode is the canonical example of how a provable theoretical "
            "limitation can stall an entire research programme."
        ),
        requires=["dl_01_perceptron"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="Minsky, M. & Papert, S. — Perceptrons: An Introduction to Computational Geometry. MIT Press (1969)",
        published_date=date(1969, 1, 1),
    ),

    # 3. Multi-Layer Perceptron (MLP)
    HistoricalEpisode(
        id="dl_03_mlp",
        concept="Multi-Layer Perceptron (MLP)",
        problem_posed=(
            "If a single layer cannot represent non-linear functions like XOR, what happens when "
            "multiple layers of threshold units are stacked? Do hidden layers add expressive power, "
            "and if so, how should the weights of hidden units — which produce no direct output "
            "error — be trained?"
        ),
        attempted_solution=(
            "Researchers including Rumelhart, Hinton, and Williams (and independently Werbos) "
            "demonstrated that stacking layers of sigmoid neurons — each computing a smooth, "
            "differentiable non-linear activation — can represent arbitrary continuous functions "
            "(Universal Approximation Theorem, Cybenko 1989). The MLP architecture arranges "
            "neurons in an input layer, one or more hidden layers, and an output layer, with "
            "fully-connected feedforward weights between consecutive layers."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "The MLP resolved the XOR failure by introducing hidden units whose intermediate "
            "representations can encode non-linear feature combinations. The expressive power of "
            "two-layer MLPs with sigmoid activations was shown to be universal — they can "
            "approximate any measurable function to arbitrary precision given enough hidden units. "
            "This architectural insight revived neural-network research and set the stage for "
            "efficient training via backpropagation."
        ),
        requires=["dl_02_xor_failure"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source=(
            "Rumelhart, D.E., Hinton, G.E. & Williams, R.J. — Learning Representations by "
            "Back-propagating Errors. Nature, 323, 533–536 (1986); "
            "Cybenko, G. — Approximation by Superpositions of a Sigmoidal Function. "
            "Mathematics of Control, Signals, and Systems, 2(4), 303–314 (1989)"
        ),
        published_date=date(1986, 1, 1),
    ),

    # 4. Backpropagation
    HistoricalEpisode(
        id="dl_04_backpropagation",
        concept="Backpropagation",
        problem_posed=(
            "An MLP has hidden layers whose weights receive no direct supervision signal — there "
            "is no target value for each hidden unit's activation. How can the credit (or blame) "
            "for an output error be efficiently distributed back through all layers so that every "
            "weight in the network can be updated by gradient descent?"
        ),
        attempted_solution=(
            "Backpropagation applies the chain rule of calculus recursively from the output layer "
            "to the input layer. For each training example, a forward pass computes all activations; "
            "then a backward pass computes the gradient of the loss with respect to every weight by "
            "propagating error deltas layer-by-layer. Each weight update is proportional to the "
            "product of the incoming activation and the backpropagated delta — an O(W) algorithm "
            "where W is the total number of weights."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Backpropagation made deep network training computationally tractable for the first "
            "time. The 1986 Rumelhart–Hinton–Williams paper popularised the algorithm (Werbos had "
            "derived it in his 1974 PhD thesis; Parker re-derived it in 1985) and demonstrated "
            "that hidden layers learn meaningful internal representations — e.g., encoding "
            "family-tree relations. Backpropagation remains the core training algorithm for "
            "virtually all neural networks today."
        ),
        requires=["dl_03_mlp"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source=(
            "Rumelhart, D.E., Hinton, G.E. & Williams, R.J. — Learning Representations by "
            "Back-propagating Errors. Nature, 323, 533–536 (1986); "
            "Werbos, P.J. — Beyond Regression: New Tools for Prediction and Analysis in the "
            "Behavioral Sciences. PhD thesis, Harvard University (1974)"
        ),
        published_date=date(1986, 9, 1),
    ),

    # 5. Convolutional Neural Network (CNN)
    HistoricalEpisode(
        id="dl_05_cnn",
        concept="Convolutional Neural Network (CNN)",
        problem_posed=(
            "Fully-connected MLPs applied to images treat every pixel as an independent input, "
            "ignoring the spatial structure of images: nearby pixels are correlated, objects can "
            "appear anywhere in the frame, and the same feature detector should fire regardless "
            "of position. How can a network exploit spatial locality and translation invariance "
            "to learn image features far more efficiently than a fully-connected architecture?"
        ),
        attempted_solution=(
            "LeCun et al. introduced the Convolutional Neural Network (1989/1998): layers of "
            "learned filters slide across the input image, each computing a local dot product "
            "(convolution) to produce a feature map. Shared weights across positions enforce "
            "translation equivariance and drastically reduce parameter count. Pooling layers "
            "sub-sample feature maps, providing translation invariance and spatial hierarchy. "
            "LeNet-5 (1998) stacked convolutional, pooling, and fully-connected layers, "
            "achieving state-of-the-art handwritten digit recognition."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "CNNs exploited the statistical structure of natural images — local connectivity, "
            "weight sharing, spatial hierarchy — to achieve parameter efficiency and strong "
            "generalisation that fully-connected networks could not match. AlexNet (2012) "
            "demonstrated that deep CNNs trained on GPUs could decisively win large-scale image "
            "classification (ImageNet), launching the modern deep learning era and making "
            "CNNs the dominant architecture for computer vision for a decade."
        ),
        requires=["dl_04_backpropagation"],
        concurrent_with=["dl_06_rnn"],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source=(
            "LeCun, Y. et al. — Gradient-Based Learning Applied to Document Recognition. "
            "Proceedings of the IEEE, 86(11), 2278–2324 (1998); "
            "Krizhevsky, A., Sutskever, I. & Hinton, G.E. — ImageNet Classification with Deep "
            "Convolutional Neural Networks. NeurIPS (2012)"
        ),
        published_date=date(1998, 11, 1),
    ),

    # 6. Recurrent Neural Network (RNN) — concurrent with CNN
    HistoricalEpisode(
        id="dl_06_rnn",
        concept="Recurrent Neural Network (RNN)",
        problem_posed=(
            "Images are static, but language, speech, and time-series data are sequential: the "
            "meaning of a word depends on what preceded it. A feedforward MLP processes each "
            "input independently with no memory of previous inputs. How can a network maintain "
            "a dynamic internal state that evolves over time, allowing it to model temporal "
            "dependencies of arbitrary length?"
        ),
        attempted_solution=(
            "Recurrent Neural Networks add directed cycles: the hidden state at each time step "
            "is a function of both the current input and the previous hidden state, creating a "
            "learned dynamic memory. Elman (1990) and Jordan (1986) proposed simple RNN "
            "architectures. Training is performed by Backpropagation Through Time (BPTT), "
            "which unrolls the recurrence into a deep feedforward graph over time steps and "
            "applies standard backpropagation."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "RNNs gave neural networks a mechanism to process variable-length sequences and "
            "capture temporal context — essential for language modelling, speech recognition, "
            "and machine translation. They were a foundational architecture for sequence "
            "modelling throughout the 1990s and early 2000s, even as practitioners struggled "
            "with training instability. The limitations of vanilla RNNs would directly motivate "
            "the invention of the LSTM."
        ),
        requires=["dl_04_backpropagation"],
        concurrent_with=["dl_05_cnn"],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source=(
            "Elman, J.L. — Finding Structure in Time. Cognitive Science, 14(2), 179–211 (1990); "
            "Werbos, P.J. — Backpropagation Through Time: What It Does and How to Do It. "
            "Proceedings of the IEEE, 78(10), 1550–1560 (1990)"
        ),
        published_date=date(1990, 1, 1),
    ),

    # 7. Vanishing Gradient Problem (FAILURE)
    HistoricalEpisode(
        id="dl_07_vanishing_gradient",
        concept="Vanishing (and exploding) gradient problem",
        problem_posed=(
            "RNNs trained with BPTT must propagate gradients backwards through many time steps. "
            "Deep feedforward networks trained with backpropagation face the same issue across "
            "many layers. In both cases, the gradient of the loss with respect to early-layer "
            "weights involves a product of many Jacobian matrices. What happens to this product "
            "as depth or sequence length grows?"
        ),
        attempted_solution=(
            "Hochreiter (1991) and Bengio et al. (1994) analysed the problem mathematically. "
            "When the spectral norm of the weight Jacobians is consistently less than 1, the "
            "gradient signal decays exponentially with depth — the vanishing gradient problem. "
            "When it is consistently greater than 1, gradients explode. Various heuristics were "
            "tried: careful weight initialisation, gradient clipping (for explosion), and "
            "truncated BPTT (reducing the unrolled depth), but none solved the fundamental "
            "long-range dependency learning problem."
        ),
        outcome=Outcome.FAILURE,
        why=(
            "The vanishing gradient problem made it practically impossible for RNNs to learn "
            "dependencies spanning more than roughly 10–20 time steps, and for deep networks to "
            "train their earliest layers effectively. It was the central obstacle blocking "
            "deep learning for over a decade. The failure to solve it with simple architectural "
            "or optimisation patches directly motivated the gated memory architecture of the LSTM."
        ),
        requires=["dl_05_cnn", "dl_06_rnn"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source=(
            "Hochreiter, S. — Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis, "
            "TU München (1991); "
            "Bengio, Y., Simard, P. & Frasconi, P. — Learning Long-Term Dependencies with "
            "Gradient Descent is Difficult. IEEE Transactions on Neural Networks, 5(2), 157–166 (1994)"
        ),
        published_date=date(1994, 1, 1),
    ),

    # 8. LSTM
    HistoricalEpisode(
        id="dl_08_lstm",
        concept="Long Short-Term Memory (LSTM)",
        problem_posed=(
            "Vanilla RNNs cannot learn long-range dependencies because gradients vanish. The "
            "root cause is that the hidden state is entirely overwritten at each step — there is "
            "no mechanism to preserve information selectively. How can a recurrent architecture "
            "be designed to maintain and access information over hundreds of time steps without "
            "gradient degradation?"
        ),
        attempted_solution=(
            "Hochreiter and Schmidhuber (1997) introduced the Long Short-Term Memory cell. The "
            "LSTM replaces the simple hidden-state update with a cell state — a conveyor belt of "
            "information — protected by three learned gating mechanisms: the forget gate (decide "
            "what to erase), the input gate (decide what to write), and the output gate (decide "
            "what to expose). Because the cell state update path has a near-constant gradient "
            "norm (the Constant Error Carousel), gradients can flow back hundreds of steps "
            "without vanishing."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "The LSTM's gating architecture provided a principled, end-to-end-differentiable "
            "solution to the vanishing gradient problem. It achieved state-of-the-art results "
            "on speech recognition, handwriting recognition, and language modelling, and became "
            "the dominant sequence model from roughly 2013 to 2017. It demonstrated that "
            "architectural inductive biases — not just optimisation tricks — were the key to "
            "training deep temporal models."
        ),
        requires=["dl_07_vanishing_gradient"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source=(
            "Hochreiter, S. & Schmidhuber, J. — Long Short-Term Memory. "
            "Neural Computation, 9(8), 1735–1780 (1997)"
        ),
        published_date=date(1997, 11, 1),
    ),

    # 9. Attention Mechanism
    HistoricalEpisode(
        id="dl_09_attention",
        concept="Attention mechanism",
        problem_posed=(
            "Even with LSTMs, encoder–decoder sequence-to-sequence models for tasks like machine "
            "translation compress an entire variable-length input sentence into a single fixed-size "
            "context vector — a bottleneck that degrades quality on long sentences. How can the "
            "decoder selectively focus on different parts of the encoder's output at each "
            "decoding step, rather than relying on a single compressed summary?"
        ),
        attempted_solution=(
            "Bahdanau, Cho, and Bengio (2014) introduced additive attention: at each decoder "
            "step, a learned alignment model scores the compatibility of the decoder's current "
            "hidden state with each encoder hidden state. The scores are normalised with a "
            "softmax to produce attention weights, and the context vector is the weighted sum "
            "of all encoder states. The alignment model is jointly trained with the full "
            "encoder–decoder system via backpropagation. Luong et al. (2015) proposed faster "
            "multiplicative (dot-product) variants."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "Attention broke the fixed-context-vector bottleneck and dramatically improved "
            "neural machine translation on long sentences. The learned alignment weights are "
            "interpretable — they show which source words the model attends to when generating "
            "each target word. More fundamentally, attention provided a differentiable, "
            "content-based memory access mechanism that could be generalised far beyond RNNs, "
            "eventually replacing recurrence entirely in the Transformer architecture."
        ),
        requires=["dl_08_lstm"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source=(
            "Bahdanau, D., Cho, K. & Bengio, Y. — Neural Machine Translation by Jointly "
            "Learning to Align and Translate. ICLR (2015) [arXiv:1409.0473]; "
            "Luong, M.-T., Pham, H. & Manning, C.D. — Effective Approaches to Attention-based "
            "Neural Machine Translation. EMNLP (2015)"
        ),
        published_date=date(2015, 1, 1),
    ),

    # 10. Transformer
    HistoricalEpisode(
        id="dl_10_transformer",
        concept="Transformer",
        problem_posed=(
            "Attention mechanisms had been grafted onto RNNs, but RNNs process tokens "
            "sequentially — each hidden state depends on the previous one, preventing "
            "parallelisation during training and limiting scalability. If attention can capture "
            "all pairwise dependencies between tokens in a sequence, is recurrence necessary "
            "at all? Can a model built entirely from attention and point-wise feedforward layers "
            "outperform recurrent architectures while being massively parallelisable?"
        ),
        attempted_solution=(
            "Vaswani et al. (2017) introduced the Transformer: an encoder–decoder architecture "
            "in which every layer is composed solely of Multi-Head Self-Attention and position-wise "
            "feedforward networks, with residual connections and layer normalisation. Self-attention "
            "computes scaled dot-product attention over all positions simultaneously — O(n²) in "
            "sequence length but fully parallelisable. Positional encodings inject sequence-order "
            "information since there are no recurrences. Multi-head attention runs multiple "
            "attention functions in parallel, capturing different types of relationships."
        ),
        outcome=Outcome.SUCCESS,
        why=(
            "'Attention Is All You Need' replaced RNNs as the dominant architecture for NLP "
            "within two years. Full parallelisation enabled training on unprecedentedly large "
            "datasets: BERT (2018), GPT-2 (2019), T5 (2019), and GPT-3 (2020) all built "
            "directly on the Transformer. The architecture generalised beyond language to vision "
            "(ViT, 2020), audio, protein structure (AlphaFold 2), and multi-modal models, "
            "making it the foundational building block of modern large foundation models."
        ),
        requires=["dl_09_attention"],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source=(
            "Vaswani, A. et al. — Attention Is All You Need. NeurIPS (2017) [arXiv:1706.03762]; "
            "Devlin, J. et al. — BERT: Pre-training of Deep Bidirectional Transformers for "
            "Language Understanding. NAACL (2019) [arXiv:1810.04805]"
        ),
        published_date=date(2017, 6, 1),
    ),
]


# ---------------------------------------------------------------------------
# Lifespan hook — seed Track A on first startup (Requirement 10.3)
# ---------------------------------------------------------------------------

import logging
import cognee
from memory.gateway import AgentRole, MemoryGateway

logger = logging.getLogger(__name__)


async def seed_tracks_if_absent() -> None:
    """
    Check whether each seed topic is already present in Track A; if not,
    write it via the Ingestion gateway and consolidate entity descriptions.

    Designed to be called once from the FastAPI lifespan hook (backend/main.py).
    Exceptions are caught and logged as warnings so a failed seed never
    prevents the application from starting.

    Satisfies Requirements 10.3, 2.4, 2.5.
    """
    gateway = MemoryGateway(role=AgentRole.INGESTION)

    # --- OS Memory Management ---
    try:
        existing_os = await cognee.recall(
            graph_name="content_track", query="OS memory management"
        )
        if not existing_os:
            logger.info("Seeding Track A: OS memory management episodes.")
            await gateway.add_data_points(OS_MEMORY_EPISODES, temporal_cognify=True)
            await cognee.consolidate_entity_descriptions_pipeline()
            logger.info("OS memory management seed complete.")
        else:
            logger.debug("Track A already contains OS memory management episodes; skipping seed.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "seed_tracks_if_absent: failed to seed OS memory management episodes: %s", exc
        )

    # --- Deep Learning ---
    try:
        existing_dl = await cognee.recall(
            graph_name="content_track", query="deep learning"
        )
        if not existing_dl:
            logger.info("Seeding Track A: deep learning episodes.")
            await gateway.add_data_points(DEEP_LEARNING_EPISODES, temporal_cognify=True)
            await cognee.consolidate_entity_descriptions_pipeline()
            logger.info("Deep learning seed complete.")
        else:
            logger.debug("Track A already contains deep learning episodes; skipping seed.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "seed_tracks_if_absent: failed to seed deep learning episodes: %s", exc
        )
