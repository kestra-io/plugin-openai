package io.kestra.plugin.openai;

import com.theokanning.openai.service.OpenAiService;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
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
public abstract class AbstractTask extends Task implements OpenAiInterface{
    protected String apiKey;

    protected String user;

    protected OpenAiService client(RunContext runContext) throws IllegalVariableEvaluationException {
        String apiKey = runContext.render(this.apiKey);

        return new OpenAiService(apiKey);
    }
}
