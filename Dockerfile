FROM python:3.13-slim AS base
WORKDIR /app

# ---- Stage 1: Install dependencies + download & compile corpora ----
# This layer is cached as long as pyproject.toml doesn't change.

COPY pyproject.toml ./
# Minimal src stub so pip install . can resolve the package
RUN mkdir -p src/context_fabric_mcp && touch src/context_fabric_mcp/__init__.py
RUN pip install --no-cache-dir .

# Also install text-fabric for corpus pre-download (build-time only)
RUN pip install --no-cache-dir "text-fabric>=12.0.0"

# Pre-download Text-Fabric corpora at build time.
ENV HOME=/root
RUN python -c "\
from tf.app import use; \
use('ETCBC/bhsa', silent='deep'); \
use('ETCBC/nestle1904', silent='deep')" \
 && test -f /root/text-fabric-data/github/ETCBC/bhsa/tf/2021/otext.tf \
 && echo '=== Corpus pre-download OK ==='

# Pre-compile .cfm caches at build time to avoid memory spike at runtime.
# Each corpus is compiled in a SEPARATE Python process so memory is freed
# between them (BHSA alone needs ~3.5 GiB during compilation).

# 1) Compile Hebrew (BHSA)
RUN python -c "\
import cfabric; \
CF = cfabric.Fabric(locations='/root/text-fabric-data/github/ETCBC/bhsa/tf/2021', silent=True); \
CF.loadAll(silent=True); \
print('Compiled: BHSA')" \
 && test -d /root/text-fabric-data/github/ETCBC/bhsa/tf/2021/.cfm \
 && echo '=== BHSA .cfm OK ==='

# 2) Compile Greek (Nestle1904) — hide nodeId.tf first (int32 overflow in CF 0.5.x)
RUN mv /root/text-fabric-data/github/ETCBC/nestle1904/tf/*/nodeId.tf \
       /root/text-fabric-data/github/ETCBC/nestle1904/tf/nodeId.tf._skip 2>/dev/null; \
    python -c "\
import cfabric; \
CF = cfabric.Fabric(locations='/root/text-fabric-data/github/ETCBC/nestle1904/tf/0.4.0', silent=True); \
CF.loadAll(silent=True); \
print('Compiled: Nestle1904')" \
 && mv /root/text-fabric-data/github/ETCBC/nestle1904/tf/nodeId.tf._skip \
       /root/text-fabric-data/github/ETCBC/nestle1904/tf/*/nodeId.tf 2>/dev/null; \
    test -d /root/text-fabric-data/github/ETCBC/nestle1904/tf/0.4.0/.cfm \
 && echo '=== Nestle1904 .cfm OK ==='

# ---- Stage 2: Copy application code (changes frequently) ----
# Only this layer rebuilds when src/ or prompt files change.

COPY src/ ./src/
RUN pip install --no-cache-dir .
COPY system_prompt.md system_prompt_quiz.md ./

# At runtime, HOME=/data (persistent volume).
# Pre-compiled .cfm caches are copied to the volume on first boot.
ENV HOME=/data
ENV QUIZ_DIR=/data/quizzes

EXPOSE 8000

CMD ["cf-api"]
