# Kestra OpenAI Plugin

## What

- Provides plugin components under `io.kestra.plugin.openai`.
- Includes classes such as `CreateImage`, `Responses`, `ChatCompletion`, `UploadFile`.

## Why

- This plugin integrates Kestra with OpenAI.
- It provides tasks that call OpenAI for chat completions, images, and file uploads.

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
