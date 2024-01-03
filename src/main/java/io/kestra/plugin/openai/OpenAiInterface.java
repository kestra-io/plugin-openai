package io.kestra.plugin.openai;

import io.kestra.core.models.annotations.PluginProperty;
import io.swagger.v3.oas.annotations.media.Schema;

import jakarta.validation.constraints.NotNull;

public interface OpenAiInterface {
    @Schema(
        title = "The OpenAI API key"
    )
    @PluginProperty(dynamic = true)
    @NotNull
    String getApiKey();

    @Schema(
        title = "A unique identifier representing your end-user."
    )
    @PluginProperty(dynamic = true)
    String getUser();

    @Schema(
        title = "The maximum number of seconds to wait for a response."
    )
    long getClientTimeout();
}
