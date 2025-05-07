package io.kestra.plugin.openai;

import io.kestra.core.models.property.Property;
import io.swagger.v3.oas.annotations.media.Schema;

import jakarta.validation.constraints.NotNull;

public interface OpenAiInterface {
    @Schema(
        title = "The OpenAI API key"
    )
    @NotNull
    Property<String> getApiKey();

    @Schema(
        title = "A unique identifier representing your end-user"
    )
    Property<String> getUser();

    @Schema(
        title = "The maximum number of seconds to wait for a response"
    )
    long getClientTimeout();
}
