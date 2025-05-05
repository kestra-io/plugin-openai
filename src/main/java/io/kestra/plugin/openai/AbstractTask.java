package io.kestra.plugin.openai;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.theokanning.openai.service.OpenAiService;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;

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
    private Property<String> apiKey;

    protected Property<String> user;

    @Builder.Default
    protected long clientTimeout = 10;

    protected OpenAiService client(RunContext runContext) throws IllegalVariableEvaluationException {
        String apiKey = runContext.render(runContext.render(this.apiKey).as(String.class).orElseThrow());

        return new OpenAiService(apiKey, Duration.ofSeconds(clientTimeout));
    }

    protected OpenAIClient openAIClient(RunContext runContext) throws IllegalVariableEvaluationException {
        String apiKey = runContext.render(runContext.render(this.apiKey).as(String.class).orElseThrow());

        return OpenAIOkHttpClient.builder().apiKey(apiKey).build();
    }
}
