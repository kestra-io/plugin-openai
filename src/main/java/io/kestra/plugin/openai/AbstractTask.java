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
    @Schema(
        title = "OpenAI API key"
    )
    @NotNull
    @PluginProperty(secret = true, group = "main")
    private Property<String> apiKey;

    protected Property<String> user;

    @Builder.Default
    @Min(1)
    protected Property<Long> clientTimeout = Property.ofValue(600L);

    protected OpenAIClient openAIClient(RunContext runContext) throws IllegalVariableEvaluationException {
        var rApiKey = runContext.render(this.apiKey).as(String.class).orElseThrow(() -> new IllegalArgumentException("apiKey is required"));
        var rClientTimeout = runContext.render(this.clientTimeout).as(Long.class).orElse(600L);

        return OpenAIOkHttpClient.builder()
            .apiKey(rApiKey)
            .timeout(Duration.ofSeconds(rClientTimeout))
            .build();
    }

    public abstract static class AbstractTaskBuilder<C extends AbstractTask, B extends AbstractTaskBuilder<C, B>> extends Task.TaskBuilder<C, B> {
        public B clientTimeout(Property<Long> clientTimeout) {
            this.clientTimeout$value = clientTimeout;
            this.clientTimeout$set = true;
            return self();
        }

        public B clientTimeout(long clientTimeout) {
            return this.clientTimeout(Property.ofValue(clientTimeout));
        }
    }
}
