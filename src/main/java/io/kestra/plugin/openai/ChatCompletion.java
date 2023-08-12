package io.kestra.plugin.openai;

import com.theokanning.openai.Usage;
import com.theokanning.openai.completion.chat.ChatCompletionChoice;
import com.theokanning.openai.completion.chat.ChatCompletionRequest;
import com.theokanning.openai.completion.chat.ChatCompletionResult;
import com.theokanning.openai.completion.chat.ChatMessage;
import com.theokanning.openai.completion.chat.ChatFunction;
import com.theokanning.openai.service.OpenAiService;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;
import lombok.experimental.SuperBuilder;

import javax.validation.constraints.NotNull;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Given a prompt, get a response from an LLM using the [OpenAI’s Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)",
    description = "For more information, refer to the [Chat Completions API docs](https://platform.openai.com/docs/guides/gpt/chat-completions-api)"
)
@Plugin(
    examples = {
        @Example(
            title = "Based on a prompt input, generate a completion response and pass it to a downstream task",
            full = true,
            code = {
                "id: openAI",
                "namespace: dev",
                "",
                "inputs:",
                "  - name: prompt",
                "    type: STRING",
                "    defaults: What is data orchestration?",
                "",
                "tasks:",
                "  - id: completion",
                "    type: io.kestra.plugin.openai.ChatCompletion",
                "    apiKey: \"yourOpenAIapiKey\"",
                "    model: gpt-3.5-turbo-0613",
                "    prompt: \"{{inputs.prompt}}\"",
                "",
                "  - id: response",
                "    type: io.kestra.core.tasks.debugs.Return",
                "    format: \"{{outputs.completion.choices[0].message.content}}\""
            }
        )
    }
)
public class ChatCompletion extends AbstractTask implements RunnableTask<ChatCompletion.Output> {
    @Schema(
        title = "A list of messages comprising the conversation so far.",
        description = "Required if prompt is not set."
    )
    @PluginProperty
    private List<ChatMessage> messages;

    @Schema(
        title = "The function call(s) the API can use when generating completions."
    )
    @PluginProperty
    private List<ChatFunction> functions;

    @Schema(
        title = "The prompt(s) to generate completions for. By default, this prompt will be sent as a `user` role.",
        description = "If not provided, make sure to set the `messages` property."
    )
    @PluginProperty
    private String prompt;

    @Schema(
        title = "What sampling temperature to use, between 0 and 2. Defaults to 1."
    )
    @PluginProperty
    private Double temperature;

    @Schema(
        title = "An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass. Defaults to 1."
    )
    @PluginProperty
    private Double topP;

    @Schema(
        title = "How many chat completion choices to generate for each input message. Defaults to 1."
    )
    private Integer n;

    @Schema(
        title = "Up to 4 sequences where the API will stop generating further tokens. Defaults to null."
    )
    @PluginProperty
    private List<String> stop;

    @Schema(
        title = "The maximum number of tokens to generate in the chat completion. No limits are set by default."
    )
    @PluginProperty
    private Integer maxTokens;

    @Schema(
        title = "Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far. Defaults to 0."
    )
    @PluginProperty
    private Double presencePenalty;

    @Schema(
        title = "Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far. Defaults to 0."
    )
    @PluginProperty
    private Double frequencyPenalty;

    @Schema(
        title = "Modify the likelihood of specified tokens appearing in the completion. Defaults to null."
    )
    @PluginProperty
    private Map<String, Integer> logitBias;

    @Schema(
        title = "ID of the model to use e.g. `'gpt-4'`",
        description = "See the [OpenAI model's documentation page](https://platform.openai.com/docs/models/overview) for more details."
    )
    @PluginProperty(dynamic = true)
    @NotNull
    private String model;

    @Override
    public ChatCompletion.Output run(RunContext runContext) throws Exception {
        OpenAiService client = this.client(runContext);

        if (this.messages == null && this.prompt == null) {
            throw new IllegalArgumentException("Either `messages` or `prompt` must be set");
        }

        List<String> stop = this.stop != null ? runContext.render(this.stop) : null;
        String user = runContext.render(this.user);
        String model = runContext.render(this.model);

        List<ChatMessage> messages = new ArrayList<>();
        // Render all messages content
        if (this.messages != null) {
            for (ChatMessage message : this.messages) {
                message.setContent(runContext.render(message.getContent()));
                messages.add(message);
            }
        }
        if (this.prompt != null) {
            messages.add(buildMessage("user", runContext.render(this.prompt)));
        }

        ChatCompletionResult chatCompletionResult = client.createChatCompletion(ChatCompletionRequest.builder()
            .messages(messages)
            .model(model)
            .temperature(this.temperature)
            .topP(this.topP)
            .n(this.n)
            .stop(stop)
            .maxTokens(this.maxTokens)
            .presencePenalty(this.presencePenalty)
            .frequencyPenalty(this.frequencyPenalty)
            .logitBias(this.logitBias)
            .user(user)
            .build()
        );

        runContext.metric(Counter.of("usage.prompt_tokens", chatCompletionResult.getUsage().getPromptTokens()));
        runContext.metric(Counter.of("usage.completion_tokens", chatCompletionResult.getUsage().getCompletionTokens()));
        runContext.metric(Counter.of("usage.total_tokens", chatCompletionResult.getUsage().getTotalTokens()));

        return ChatCompletion.Output.builder()
            .id(chatCompletionResult.getId())
            .object(chatCompletionResult.getObject())
            .model(chatCompletionResult.getModel())
            .choices(chatCompletionResult.getChoices())
            .usage(chatCompletionResult.getUsage())
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(
            title = "Unique ID assigned to this Chat Completion."
        )
        private String id;

        @Schema(
            title = "The type of object returned, should be \"chat.completion\"."
        )
        private String object;

        @Schema(
            title="The GPT model used."
        )
        private String model;

        @Schema(
            title = "A list of all generated completions."
        )
        private List<ChatCompletionChoice> choices;

        @Schema(
            title = "The API usage for this request."
        )
        private Usage usage;
    }

    private ChatMessage buildMessage(String role, String content) {
        ChatMessage message = new ChatMessage();
        message.setRole(role);
        message.setContent(content);

        return message;
    }
}