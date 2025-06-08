package io.kestra.plugin.openai;

import com.openai.client.OpenAIClient;
import com.openai.core.JsonValue;
import com.openai.models.FunctionDefinition;
import com.openai.models.FunctionParameters;
import com.openai.models.chat.completions.ChatCompletionAssistantMessageParam;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionMessageParam;
import com.openai.models.chat.completions.ChatCompletionSystemMessageParam;
import com.openai.models.chat.completions.ChatCompletionTool;
import com.openai.models.chat.completions.ChatCompletionToolChoiceOption;
import com.openai.models.chat.completions.ChatCompletionUserMessageParam;
import com.openai.models.completions.CompletionUsage;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Given a prompt, get a response from an LLM using the OpenAI’s Chat Completions API.",
    description = "For more information, refer to the [Chat Completions API docs](https://platform.openai.com/docs/guides/gpt/chat-completions-api)."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Based on a prompt input, generate a completion response and pass it to a downstream task.",
            code = """
                id: openai_chat
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: What is data orchestration?

                tasks:
                  - id: completion
                    type: io.kestra.plugin.openai.ChatCompletion
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    model: gpt-4o
                    prompt: "{{ inputs.prompt }}"

                  - id: log_output
                    type: io.kestra.plugin.core.log.Log
                    message: "{{ outputs.completion.choices[0].message.content }}"
                """
        ),
        @Example(
            full = true,
            title = "Send a prompt to OpenAI's ChatCompletion API.",
            code = """
                id: openai_chat
                namespace: company.team

                tasks:
                  - id: prompt
                    type: io.kestra.plugin.openai.ChatCompletion
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    model: gpt-4o
                    prompt: Explain in one sentence why data engineers build data pipelines

                  - id: use_output
                    type: io.kestra.plugin.core.log.Log
                    message: "{{ outputs.prompt.choices | jq('.[].message.content') | first }}"
                """
        ),
        @Example(
            full = true,
            title = "Based on a prompt input, ask OpenAI to call a function that determines whether you need to " +
                "respond to a customer's review immediately or wait until later, and then comes up with a " +
                "suggested response.",
            code = """
                id: openai_chat
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: I love your product and would purchase it again!

                tasks:
                  - id: prioritize_response
                    type: io.kestra.plugin.openai.ChatCompletion
                    apiKey: "yourOpenAIapiKey"
                    model: gpt-4o
                    messages:
                      - role: user
                        content: "{{ inputs.prompt }}"
                    functions:
                      - name: respond_to_review
                        description: Given the customer product review provided as input, determines how urgently a reply is required and then provides suggested response text.
                        parameters:
                          - name: response_urgency
                            type: string
                            description: How urgently this customer review needs a reply. Bad reviews
                                         must be addressed immediately before anyone sees them. Good reviews can
                                         wait until later.
                            required: true
                            enumValues:
                              - reply_immediately
                              - reply_later
                          - name: response_text
                            type: string
                            description: The text to post online in response to this review.
                            required: true

                  - id: response_urgency
                    type: io.kestra.plugin.core.debug.Return
                    format: "{{ outputs.prioritize_response.choices[0].message.function_call.arguments.response_urgency }}"

                  - id: response_text
                    type: io.kestra.plugin.core.debug.Return
                    format: "{{ outputs.prioritize_response.choices[0].message.function_call.arguments.response_text }}"
                """
        )
    }
)
public class ChatCompletion extends AbstractTask implements RunnableTask<ChatCompletion.Output> {
    @Schema(
        title = "A list of messages comprising the conversation so far",
        description = "This property is required if prompt is not set."
    )
    private Property<List<ChatMessage>> messages;

    @Schema(
        title = "The function call(s) the API can use when generating completions."
    )
    private Property<List<PluginChatFunction>> functions;

    @Schema(
        title = "The name of the function OpenAI should generate a call for.",
        description = "Enter a specific function name, or 'auto' to let the model decide. The default is auto."
    )
    private Property<String> functionCall;

    @Schema(
        title = "The prompt(s) to generate completions for. By default, this prompt will be sent as a `user` role.",
        description = "If not provided, make sure to set the `messages` property."
    )
    private Property<String> prompt;

    @Schema(
        title = "What sampling temperature to use, between 0 and 2. Defaults to 1."
    )
    private Property<Double> temperature;

    @Schema(
        title = "An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass. Defaults to 1."
    )
    private Property<Double> topP;

    @Schema(
        title = "How many chat completion choices to generate for each input message. Defaults to 1."
    )
    private Property<Integer> n;

    @Schema(
        title = "Up to 4 sequences where the API will stop generating further tokens. Defaults to null."
    )
    private Property<List<String>> stop;

    @Schema(
        title = "The maximum number of tokens to generate in the chat completion. No limits are set by default."
    )
    private Property<Long> maxTokens;

    @Schema(
        title = "Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far. Defaults to 0."
    )
    private Property<Double> presencePenalty;

    @Schema(
        title = "Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far. Defaults to 0."
    )
    private Property<Double> frequencyPenalty;

    @Schema(
        title = "Modify the likelihood of specified tokens appearing in the completion. Defaults to null."
    )
    private Property<Map<String, Integer>> logitBias;

    @Schema(
        title = "ID of the model to use e.g. `'gpt-4'`",
        description = "See the [OpenAI model's documentation page](https://platform.openai.com/docs/models/overview) for more details."
    )
    @NotNull
    private Property<String> model;
    @Override
    public ChatCompletion.Output run(RunContext runContext) throws Exception {
        OpenAIClient client = this.openAIClient(runContext);

        if (this.messages == null && this.prompt == null) {
            throw new IllegalArgumentException("Either `messages` or `prompt` must be set");
        }

        List<String> stop = this.stop != null ? runContext.render(this.stop).asList(String.class) : Collections.emptyList();
        String user = this.user == null ? null : runContext.render(this.user).as(String.class).orElseThrow();
        String model = this.model == null ? null : runContext.render(this.model).as(String.class).orElseThrow();

        List<ChatCompletionMessageParam> messages = new ArrayList<>();
        // Render all messages content
        if (this.messages != null) {
            for (ChatMessage message : runContext.render(this.messages).asList(ChatMessage.class)) {
                messages.add(buildMessage(message.getRole(), message.getContent()));
            }
        }

        if (this.prompt != null) {
            messages.add(buildMessage("user", runContext.render(this.prompt).as(String.class).orElseThrow()));
        }

        List<ChatCompletionTool> chatFunctions = null;
        if (this.functions != null) {
            chatFunctions = new ArrayList<>();
            for (PluginChatFunction function : runContext.render(functions).asList(PluginChatFunction.class)) {
                if(function.parameters != null) {
                    chatFunctions.add(ChatCompletionTool.builder()
                        .function(toFunctionDefinition(runContext, function))
                    .build());
                }
            }
        }
        ChatCompletionToolChoiceOption chatFunctionCall = null;

        if (this.functionCall != null) {
            var toolName = runContext.render(this.functionCall)
                .as(String.class)
                .orElse(ChatCompletionToolChoiceOption.Auto.AUTO.asString());
            chatFunctionCall = ChatCompletionToolChoiceOption.ofAuto(ChatCompletionToolChoiceOption.Auto.of(toolName));
        } else {
            chatFunctionCall = ChatCompletionToolChoiceOption.ofAuto(ChatCompletionToolChoiceOption.Auto.AUTO);
        }

        ChatCompletionCreateParams.Builder builder = ChatCompletionCreateParams.builder()
            .messages(messages)
            .model(model)
            .toolChoice(chatFunctionCall)
            .temperature(this.temperature == null ? null : runContext.render(this.temperature).as(Double.class).orElseThrow())
            .topP(this.topP == null ? null : runContext.render(this.topP).as(Double.class).orElseThrow())
            .n(this.n == null ? 1 : runContext.render(this.n).as(Integer.class).orElseThrow())
            .maxCompletionTokens(this.maxTokens == null ? null : runContext.render(this.maxTokens).as(Long.class).orElseThrow())
            .presencePenalty(this.presencePenalty == null ? null : runContext.render(this.presencePenalty).as(Double.class).orElseThrow())
            .frequencyPenalty(this.frequencyPenalty == null ? null : runContext.render(this.frequencyPenalty).as(Double.class).orElseThrow());
        Optional.ofNullable(chatFunctions).ifPresent(builder::tools);
        Optional.ofNullable(user).ifPresent(builder::user);
        Optional.ofNullable(stop).ifPresent(e-> builder.stop(ChatCompletionCreateParams.Stop.ofStrings(stop)));
        if (this.logitBias != null) {
            final Map<String, Integer> logitBias = runContext.render(this.logitBias).asMap(String.class, Integer.class);
            if (!logitBias.isEmpty()) {
                builder.logitBias(ChatCompletionCreateParams.LogitBias.builder().putAdditionalProperty(PROPERTIES, JsonValue.from(logitBias)).build());
            }
        }
        final ChatCompletionCreateParams chatCompletionCreateParams = builder.build();
        com.openai.models.chat.completions.ChatCompletion chatCompletionResult = client.chat().completions()
            .create(chatCompletionCreateParams);

        if (chatCompletionResult.usage().isPresent()) {
            runContext.metric(Counter.of("usage.prompt.tokens", chatCompletionResult.usage().get().promptTokens()));
            runContext.metric(Counter.of("usage.completion.tokens", chatCompletionResult.usage().get().completionTokens()));
            runContext.metric(Counter.of("usage.total.tokens", chatCompletionResult.usage().get().totalTokens()));
        }
        return ChatCompletion.Output.builder()
            .id(chatCompletionResult.id())
            .object(String.valueOf(chatCompletionResult._object_()))
            .model(chatCompletionResult.model())
            .choices(chatCompletionResult.choices())
            .usage(chatCompletionResult.usage().isPresent() ? chatCompletionResult.usage().get() : null)
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(
            title = "Unique ID assigned to this Chat Completion"
        )
        private String id;

        @Schema(
            title = "The type of object returned, should be \"chat.completion\"."
        )
        private String object;

        @Schema(
            title="The GPT model used"
        )
        private String model;

        @Schema(
            title = "A list of all generated completions"
        )
        private List<com.openai.models.chat.completions.ChatCompletion.Choice> choices;

        @Schema(
            title = "The API usage for this request"
        )
        private CompletionUsage usage;
    }

    @Builder
    @Getter
    public static class PluginChatFunctionParameter {
        @Schema(
            title = "The name of the function parameter"
        )
        @NotNull
        private Property<String> name;

        @Schema(
            title = "A description of the function parameter",
            description = "Provide as many details as possible to ensure the model returns an accurate parameter."
        )
        @NotNull
        private Property<String> description;

        @Schema(
            title = "The OpenAPI data type of the parameter",
            description = "Valid types are string, number, integer, boolean, array, object"
        )
        @NotNull
        private Property<String> type;

        @Schema(
            title = "A list of values that the model *must* choose from when setting this parameter.",
            description = "Optional, but useful when for classification problems."
        )
        private Property<List<String>> enumValues;

        @Schema(
            title = "Whether or not the model is required to provide this parameter",
            description = "Defaults to false."
        )
        private Property<Boolean> required;
    }

    @Builder
    @Getter
    public static class PluginChatFunction {
        @Schema(
            title = "The name of the function"
        )
        private Property<String> name;

        @Schema(
            title = "A description of what the function does"
        )
        private Property<String> description;

        @Schema(
            title = "The function's parameters"
        )
        private Property<List<PluginChatFunctionParameter>> parameters;
    }
    @Builder
    @Getter
    public static class ChatMessage {
        @NonNull
        String role;
        String content;
        String name;
    }
    private ChatCompletionMessageParam buildMessage(String role, String content) {
        return switch (Role.fromString(role)) {
            case ASSISTANT -> ChatCompletionMessageParam.ofAssistant(
                ChatCompletionAssistantMessageParam.builder().content(content).build()
            );
            case SYSTEM -> ChatCompletionMessageParam.ofSystem(
                ChatCompletionSystemMessageParam.builder().content(content).build()
            );
            case USER -> ChatCompletionMessageParam.ofUser(
                ChatCompletionUserMessageParam.builder().content(content).build()
            );
        };
    }
    private enum Role {
        ASSISTANT, SYSTEM, USER;
        private static Role fromString(final String role) {
            return switch (role.toLowerCase()) {
                case "assistant" -> ASSISTANT;
                case "system" -> SYSTEM;
                default -> USER;
            };
        }
    }
    private static final String TYPE = "type";
    private static final String ENUM = "enum";
    private static final String PROPERTIES = "properties";
    private static final String REQUIRED = "required";
    private static final String PARAMETERS = "parameters";
    private static final String DESCRIPTIONS = "description";
    private static final String STRING = "string";
    private static final String OBJECT = "object";
    private static FunctionDefinition toFunctionDefinition(final RunContext runContext,final PluginChatFunction function) throws RuntimeException, IllegalVariableEvaluationException {
        final Map<String, Object> functionParameters = new HashMap<>();
        final List<String> requiredList = new ArrayList<>();

        if (function.parameters != null) {
                for (PluginChatFunctionParameter parameter : runContext.render(function.parameters).asList(PluginChatFunctionParameter.class)) {
                    if (runContext.render(parameter.required).as(Boolean.class).orElse(false)) {
                        requiredList.add(parameter.name.toString());
                    }
                    functionParameters.put(parameter.name.toString(), Map.of(
                        TYPE, STRING,
                        DESCRIPTIONS, parameter.description,
                        ENUM, Optional.ofNullable(runContext.render(parameter.enumValues).asList(String.class)).orElse(Collections.emptyList())
                    ));
                }
        }
        return FunctionDefinition.builder()
            .name(runContext.render(function.name).as(String.class).orElseThrow())
            .description(runContext.render(function.description).as(String.class).orElseThrow())
            .parameters(FunctionParameters.builder().putAdditionalProperty(PARAMETERS, JsonValue.from(Map.of(
                TYPE, OBJECT,
                PROPERTIES, functionParameters,
                REQUIRED, requiredList.isEmpty() ? List.of() : requiredList
            ))).build())
            .build();
    }
}
