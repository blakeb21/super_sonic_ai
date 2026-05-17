# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SuperSonicAI is a Virginia Tech graduate capstone project website documenting the team's journey developing a deep reinforcement learning agent to play Sonic the Hedgehog on Sega Genesis. The site walks through each development phase as interactive documentation.

## Commands

```bash
bun run dev       # Start dev server (preferred package manager)
bun run build     # Build for production
bun run lint      # Run ESLint
bun start         # Start production server
```

No test suite exists — lint is the only automated check.

## Architecture

**Stack**: Next.js 15 + TypeScript + Tailwind CSS + PostHog analytics (no database, no API routes)

**Key directories:**
- `src/pages/` — One file per phase; no dynamic routing
- `src/components/` — Shared UI (currently just `header.tsx`)
- `src/env/` — Zod-based env validation (`schema.mjs`, `server.mjs`, `client.mjs`); only enforces `NODE_ENV`

**Page pattern**: Every phase page follows the same structure — `<Header />`, one or more `<article>` sections with prose and `<SyntaxHighlighter>` code blocks, then prev/next `<Link>` navigation buttons at the bottom. `react-syntax-highlighter` with the `vscDarkPlus` theme is used for all code samples.

**Linear narrative flow:**
Home → About → Initialization → Model Experimentation → Image Processing → Semantic Segmentation → Dataset Creation → Deep Q-Learning → DeepQ Tuning → Actor-Critic → NEAT → Model Development → Model Training → Replay Agent → Generalization → Final Product

**Styling**: Tailwind CSS with `prettier-plugin-tailwindcss` for class sorting. Global base styles in `src/styles/globals.css` apply `text-gray-100 bg-gray-800`. Prettier: single quotes, semicolons, trailing commas, print width 100, tab width 2.

**Analytics**: PostHog is initialized in `_app.tsx` by reading `NEXT_PUBLIC_POSTHOG_KEY` and `NEXT_PUBLIC_POSTHOG_HOST` directly from `process.env` (not through the Zod schema). Without `NEXT_PUBLIC_POSTHOG_KEY` set, analytics silently no-ops.
