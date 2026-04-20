# Deployment Guide

## LLM API Keys (for chat)

The `/api/chat` and `/api/chat-quiz` endpoints use **Groq** (Llama 3.3 70B) as the primary
provider and optionally fall back to **OpenAI** (gpt-4o-mini) on rate-limit / connection /
5xx errors from Groq.

### Groq (primary, free tier)

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up (no credit card required)
3. Create an API key
4. Set it as `GROQ_API_KEY` in your deployment environment

Groq's free tier is rate-limited per minute/day rather than capped cumulatively — see
[console.groq.com/settings/limits](https://console.groq.com/settings/limits) for the
current numbers. If traffic fits under the limits, usage is free indefinitely.

### OpenAI (optional fallback)

If Groq is rate-limited or temporarily unavailable, the server can transparently fall
back to OpenAI. This is pay-as-you-go (no free tier), so a daily cap is enforced to
prevent surprise bills.

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create an API key and add billing
3. Set it as `OPENAI_API_KEY`
4. Optionally tune `OPENAI_FALLBACK_DAILY_LIMIT` (default: 50 requests/day)

Leave `OPENAI_API_KEY` unset to disable fallback entirely — Groq failures will then
surface as chat errors to the client.

---

## Backend: context-fabric-mcp

### Local Development

```bash
cp .env.example .env   # configure API keys

# Start the API server
uv run cf-api

# Run tests
uv run pytest
```

The API runs at `http://localhost:8000`. First request loads corpora into memory (~2s with cached data).

### Deploy to Fly.io

#### Install CLI and log in

```bash
brew install flyctl        # macOS
fly auth login
```

#### Launch

```bash
fly launch
```

Edit the generated `fly.toml`:

```toml
app = "context-fabric-mcp"
primary_region = "ams"

[build]

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = "suspend"
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  memory = "1gb"
  cpu_kind = "shared"
  cpus = 1
```

#### Create persistent volume

```bash
fly volumes create app_data --size 1 --region ams
```

Add to `fly.toml`:

```toml
[[mounts]]
  source = "app_data"
  destination = "/data"
```

#### Set secrets

```bash
fly secrets set API_KEY=your-shared-secret-here           # recommended, locks down the API
fly secrets set GROQ_API_KEY=your-groq-key-here           # primary chat provider (free tier)
fly secrets set OPENAI_API_KEY=your-openai-key-here       # optional fallback (pay-as-you-go)
fly secrets set OPENAI_FALLBACK_DAILY_LIMIT=50            # optional safety cap on fallback
```

#### Deploy

```bash
fly deploy
```

#### Pre-warm corpus cache

After first deploy, make a request to each corpus to trigger Context-Fabric's `.cfm` compilation (one-time, ~10 min for BHSA):

```bash
curl https://your-app.fly.dev/api/books?corpus=hebrew
curl https://your-app.fly.dev/api/books?corpus=greek
```

#### Health check

Add to `fly.toml`:

```toml
[[services.http_checks]]
  interval = 30000
  timeout = 5000
  path = "/health"
```

### Deploy to Railway

#### Setup

1. Go to [railway.app](https://railway.app) and sign up
2. Create new project > Deploy from GitHub Repo > select `context-fabric-mcp`
3. Railway auto-detects the Dockerfile

#### Configure

In the Railway dashboard:

**Networking:** Click "Generate Domain" for a public URL.

**Volume:** Add one:

| Name | Mount Path | Size |
|------|-----------|------|
| app-data | /data | 1 GB |

**Variables:**

```
PORT=8000
API_KEY=your-shared-secret-here           # recommended, locks down the API
GROQ_API_KEY=your-groq-key-here           # primary chat provider (free tier)
OPENAI_API_KEY=your-openai-key-here       # optional fallback (pay-as-you-go)
OPENAI_FALLBACK_DAILY_LIMIT=50            # optional safety cap on fallback
```

#### Deploy

Push to GitHub — Railway auto-deploys. Or use CLI:

```bash
npm install -g @railway/cli
railway login
railway up
```

---

## Frontend Integration

The API is protected by an `API_KEY`. A frontend deployed on Netlify should use an edge function to proxy `/api/*` requests and inject the key server-side. See the frontend repository for deployment instructions.
