package io.kestra.plugin.openai;

import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

public interface OpenAiInterface {
    @Schema(
        title = "The OpenAI API key"
    )
    @NotNull
    @PluginProperty(secret = true, group = "main")
    Property<String> getApiKey();

    @Schema(
        title = "A unique identifier representing your end-user"
    )
    @PluginProperty(group = "advanced")
    Property<String> getUser();

    @Schema(
        title = "The maximum number of seconds to wait for a response",
        description = """
            The maximum time to wait for API responses, in seconds (minimum 1, default 600 seconds / 10 minutes).
            Applies per HTTP attempt. The underlying SDK retries twice by default, so a task can wait up to approximately 3 times this value before timing out.
            """
    )
    @PluginProperty(group = "execution")
    @Min(1)
    Property<Long> getClientTimeout();
}
