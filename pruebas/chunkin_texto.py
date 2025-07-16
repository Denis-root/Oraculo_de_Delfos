from langchain.text_splitter import RecursiveCharacterTextSplitter




text = """
How to split text based on semantic similarity
Taken from Greg Kamradt's wonderful notebook: 5_Levels_Of_Text_Splitting

All credit to him.
This guide covers how to split chunks based on their semantic similarity. If embeddings are sufficiently far apart, chunks are split.
At a high level, this splits into sentences, then groups into groups of 3 sentences, and then merges one that are similar in the embedding space.


Breakpoints
This chunker works by determining when to "break" apart sentences. This is done by looking for differences in embeddings between any two sentences. When that difference is past some threshold, then they are split.
There are a few ways to determine what that threshold is, which are controlled by the breakpoint_threshold_type kwarg.
Note: if the resulting chunk sizes are too small/big, the additional kwargs breakpoint_threshold_amount and min_chunk_size can be used for adjustments.

Percentile
The default way to split is based on percentile. In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split. The default value for X is 95.0 and can be adjusted by the keyword argument breakpoint_threshold_amount which expects a number between 0.0 and 100.0.

"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # tamaño máximo de cada fragmento
    chunk_overlap=200,  # sin solapamiento
    separators=["\n\n", "\n", ".", " "]  # primero intenta por párrafo, luego línea, luego punto, luego espacio
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"[Chunk {i+1}]\n{chunk}\n---")