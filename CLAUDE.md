# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SuperSonicAI is a Virginia Tech graduate capstone project website built with the T3 Stack. It documents the team's journey developing a deep reinforcement learning agent to play Sonic the Hedgehog on Sega Genesis. The site is structured as interactive documentation walking through each development phase.

## Commands

```bash
# Development
bun run dev       # Start dev server (preferred package manager)
bun run build     # Build for production
bun run lint      # Run ESLint
bun start         # Start production server
```

## Architecture

**Stack**: Next.js 15 + TypeScript + Tailwind CSS + PostHog analytics

**Key directories:**
- `src/pages/` — All 19 pages; each page corresponds to a project phase or topic
- `src/components/` — Shared UI (currently just `header.tsx`)
- `src/env/` — Zod-based environment variable validation (`schema.mjs`, `server.mjs`, `client.mjs`)

**Page structure**: The site has a linear narrative flow through these phases:
- Home → About → Initialization → Model Experimentation → Image Processing → Semantic Segmentation → Dataset Creation → Deep Q-Learning → DeepQ Tuning → Actor-Critic → NEAT → Model Development → Model Training → Replay Agent → Generalization → Final Product

**Styling**: Tailwind CSS with the Prettier Tailwind plugin for class sorting. Dark-themed UI. Prettier settings: single quotes, semi-colons, trailing commas, print width 100, tab width 2.

**Analytics**: PostHog is integrated for product analytics.

**Environment**: The env validation in `src/env/` only enforces `NODE_ENV` — no database or external service vars are required.
