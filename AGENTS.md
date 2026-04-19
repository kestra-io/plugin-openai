# Kestra OpenAI Plugin

## What

- Provides plugin components under `io.kestra.plugin.openai`.
- Includes classes such as `CreateImage`, `Responses`, `ChatCompletion`, `UploadFile`.

## Why

- What user problem does this solve? Teams need to call OpenAI for chat completions, images, and file uploads from orchestrated workflows instead of relying on manual console work, ad hoc scripts, or disconnected schedulers.
- Why would a team adopt this plugin in a workflow? It keeps OpenAI steps in the same Kestra flow as upstream preparation, approvals, retries, notifications, and downstream systems.
- What operational/business outcome does it enable? It reduces manual handoffs and fragmented tooling while improving reliability, traceability, and delivery speed for processes that depend on OpenAI.

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

## References

- https://kestra.io/docs/plugin-developer-guide
- https://kestra.io/docs/plugin-developer-guide/contribution-guidelines
