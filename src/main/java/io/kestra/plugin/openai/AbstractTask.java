package io.kestra.plugin.openai;

import com.theokanning.openai.service.OpenAiService;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import lombok.*;
import lombok.experimental.SuperBuilder;

import java.time.Duration;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
public abstract class AbstractTask extends Task implements OpenAiInterface{
    protected Property<String> apiKey;

    protected Property<String> user;

    @Builder.Default
    protected long clientTimeout = 10;

    protected OpenAiService client(RunContext runContext) throws IllegalVariableEvaluationException {
        String apiKey = runContext.render(this.apiKey.as(runContext, String.class));

        return new OpenAiService(apiKey, Duration.ofSeconds(clientTimeout));
    }
}
