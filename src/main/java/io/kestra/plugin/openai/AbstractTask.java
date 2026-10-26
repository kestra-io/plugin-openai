package io.kestra.plugin.openai;

import java.time.Duration;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
public abstract class AbstractTask extends Task implements OpenAiInterface {
    private static final long DEFAULT_CLIENT_TIMEOUT = 600L;

    @Schema(
        title = "OpenAI API key"
    )
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> apiKey;

    protected Property<String> user;

    @Builder.Default
    protected Property<@Min(1) Long> clientTimeout = Property.ofValue(DEFAULT_CLIENT_TIMEOUT);

    protected OpenAIClient openAIClient(RunContext runContext) throws IllegalVariableEvaluationException {
        var rApiKey = runContext.render(this.apiKey).as(String.class).orElseThrow(() -> new IllegalArgumentException("apiKey is required, set it or reference a secret"));
        var rClientTimeout = runContext.render(this.clientTimeout).as(Long.class).orElse(DEFAULT_CLIENT_TIMEOUT);

        return OpenAIOkHttpClient.builder()
            .apiKey(rApiKey)
            .timeout(Duration.ofSeconds(rClientTimeout))
            .build();
    }
}
