# Kestra OpenAI Plugin

## What

Interact with OpenAI from Kestra. Exposes 4 plugin components (tasks, triggers, and/or conditions).

## Why

Enables Kestra workflows to interact with OpenAI, allowing orchestration of OpenAI-based operations as part of data pipelines and automation workflows.

## How

### Architecture

Single-module plugin. Source packages under `io.kestra.plugin`:

- `openai`

### Key Plugin Classes

- `io.kestra.plugin.openai.ChatCompletion`
- `io.kestra.plugin.openai.CreateImage`
- `io.kestra.plugin.openai.Responses`
- `io.kestra.plugin.openai.UploadFile`

### Project Structure

```
plugin-openai/
├── src/main/java/io/kestra/plugin/openai/utils/
├── src/test/java/io/kestra/plugin/openai/utils/
├── build.gradle
└── README.md
```

### Important Commands

```bash
# Build the plugin
./gradlew shadowJar

# Run tests
./gradlew test

# Build without tests
./gradlew shadowJar -x test
```

### Configuration

All tasks and triggers accept standard Kestra plugin properties. Credentials should use
`{{ secret('SECRET_NAME') }}` — never hardcode real values.

## Agents

**IMPORTANT:** This is a Kestra plugin repository (prefixed by `plugin-`, `storage-`, or `secret-`). You **MUST** delegate all coding tasks to the `kestra-plugin-developer` agent. Do NOT implement code changes directly — always use this agent.
