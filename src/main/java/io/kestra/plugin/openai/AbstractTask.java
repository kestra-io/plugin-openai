package io.kestra.plugin.openai;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.theokanning.openai.client.OpenAiApi;
import com.theokanning.openai.service.OpenAiService;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;
import okhttp3.OkHttpClient;
import retrofit2.Retrofit;

import java.time.Duration;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
public abstract class AbstractTask extends Task implements OpenAiInterface{
    protected String apiKey;

    protected String user;

    @Schema(
        title = "The timeout value to send to the OpenAI API."
    )
    @PluginProperty
    private Integer apiTimeout;

    protected OpenAiService client(RunContext runContext) throws IllegalVariableEvaluationException {
        String apiKey = runContext.render(this.apiKey);

        return createOpenAiService(apiKey, Duration.ofSeconds(apiTimeout));
    }

    private static OpenAiService createOpenAiService(String token, final Duration timeout) {
        ObjectMapper mapper = OpenAiService.defaultObjectMapper();
        mapper.addMixIn(ChatCompletion.PluginChatFunction.class, KestraChatFunctionMixin.class);

        OkHttpClient client = OpenAiService.defaultClient(token, timeout);
        Retrofit retrofit = OpenAiService.defaultRetrofit(client, mapper);

        OpenAiApi api = retrofit.create(OpenAiApi.class);
        return new OpenAiService(api);
    }

    public abstract static class KestraChatFunctionMixin {
        @JsonIgnore
        abstract Class<?> getParametersClass();

        @JsonProperty("parameters")
        abstract ArrayNode getParameters();
    }
}
