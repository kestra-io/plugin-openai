package io.kestra.plugin.openai;

import com.openai.client.OpenAIClient;
import com.openai.core.JsonValue;
import com.openai.models.FunctionDefinition;
import com.openai.models.ResponseFormatJsonSchema;
import com.openai.models.chat.completions.*;
import com.openai.models.completions.CompletionUsage;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

import java.io.IOException;
import java.util.*;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Send prompts to OpenAI Chat Completions",
    description = "Calls the OpenAI [Chat Completions API docs](https://platform.openai.com/docs/guides/gpt/chat-completions-api) with a prompt or message list, optional tool calls, and optional JSON Schema structured output. Requires either `prompt` or `messages`; defaults to OpenAI sampling limits and records token usage metrics."
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
                    format: "{{ outputs.prioritize_response | jq('.choices[0].message.tool_calls[0].function.arguments | fromjson | .response_urgency') | first }}"

                  - id: response_text
                    type: io.kestra.plugin.core.debug.Return
                    format: "{{ outputs.prioritize_response | jq('.choices[0].message.tool_calls[0].function.arguments | fromjson | .response_text') | first }}"
                """
        )
    }
)
public class ChatCompletion extends AbstractTask implements RunnableTask<ChatCompletion.Output> {
    @Schema(
        title = "Messages to send to the model",
        description = "Required when `prompt` is absent; each entry keeps its role and content."
    )
    private Property<List<ChatMessage>> messages;

    @Schema(
        title = "Functions exposed as tools",
        description = "Define tool-callable functions with names, descriptions, and parameters; ignored if none provided."
    )
    private Property<List<PluginChatFunction>> functions;

    @Schema(
        title = "Tool choice directive",
        description = "Use 'auto' (default), 'none', or a specific function name present in `functions`; other names raise an error."
    )
    @Builder.Default
    private Property<String> functionCall = Property.ofValue("auto");

    @Schema(
        title = "Prompt text to send",
        description = "Sent as a `user` message when `messages` are omitted; required if `messages` is not set."
    )
    private Property<String> prompt;

    @Schema(
        title = "Sampling temperature (0–2)",
        description = "Default 1.0; higher values increase randomness."
    )
    @Builder.Default
    private Property<Double> temperature = Property.ofValue(1.0);

    @Schema(
        title = "Nucleus sampling top_p",
        description = "Default 1.0; lower values limit token candidates."
    )
    @Builder.Default
    private Property<Double> topP = Property.ofValue(1.0);

    @Schema(
        title = "Number of choices",
        description = "Default 1; controls size of `choices` list."
    )
    @Builder.Default
    private Property<Integer> n = Property.ofValue(1);

    @Schema(
        title = "Stop sequences",
        description = "Up to 4 strings; default is none."
    )
    private Property<List<String>> stop;

    @Schema(
        title = "Max completion tokens",
        description = "Leave null to rely on model defaults; counts only completion tokens."
    )
    private Property<Long> maxTokens;

    @Schema(
        title = "Presence penalty (-2 to 2)",
        description = "Default 0; positive values discourage repeating earlier topics."
    )
    private Property<Double> presencePenalty;

    @Schema(
        title = "Frequency penalty (-2 to 2)",
        description = "Default 0; positive values reduce reuse of frequent tokens."
    )
    private Property<Double> frequencyPenalty;

    @Schema(
        title = "Token logit bias",
        description = "Map token IDs to bias values; empty when unset."
    )
    private Property<Map<String, Integer>> logitBias;

    @Schema(
        title = "Model ID",
        description = "Required OpenAI model identifier (e.g., `gpt-4o`); see the [model docs](https://platform.openai.com/docs/models/overview)."
    )
    @NotNull
    private Property<String> model;

    @Schema(
        title = "JSON response schema",
        description = "Stringified JSON Schema enabling `response_format` = `json_schema`; uses name `kestra_schema` with `strict` true."
    )
    private Property<String> jsonResponseSchema;

    @Override
    public ChatCompletion.Output run(RunContext runContext) throws Exception {
        OpenAIClient client = this.openAIClient(runContext);

        if (this.messages == null && this.prompt == null) {
            throw new IllegalArgumentException("Either `messages` or `prompt` must be set");
        }

        List<String> stop = this.stop != null ? runContext.render(this.stop).asList(String.class) : Collections.emptyList();
        String user = this.user == null ? null : runContext.render(this.user).as(String.class).orElseThrow();
        String model = this.model == null ? null : runContext.render(this.model).as(String.class).orElseThrow();
        var rJsonResponseSchema = runContext.render(this.jsonResponseSchema).as(String.class).orElse(null);

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
                if (function.parameters != null) {
                    chatFunctions.add(ChatCompletionTool.ofFunction(ChatCompletionFunctionTool.builder()
                        .function(toFunctionDefinition(runContext, function))
                        .build()));
                }
            }
        }

        ChatCompletionCreateParams.Builder builder = ChatCompletionCreateParams.builder()
            .messages(messages)
            .model(model)
            .temperature(this.temperature == null ? null : runContext.render(this.temperature).as(Double.class).orElse(1.0))
            .topP(this.topP == null ? null : runContext.render(this.topP).as(Double.class).orElse(1.0))
            .n(runContext.render(this.n).as(Integer.class).orElse(1))
            .maxCompletionTokens(this.maxTokens == null ? null : runContext.render(this.maxTokens).as(Long.class).orElseThrow())
            .presencePenalty(this.presencePenalty == null ? null : runContext.render(this.presencePenalty).as(Double.class).orElseThrow())
            .frequencyPenalty(this.frequencyPenalty == null ? null : runContext.render(this.frequencyPenalty).as(Double.class).orElseThrow());

        if (rJsonResponseSchema != null) {
            builder.responseFormat(buildOpenAIResponseFormat(rJsonResponseSchema));
        }

        String renderedFunctionCall = this.functionCall != null ? runContext.render(this.functionCall).as(String.class).orElse("auto") : "auto";

        if (chatFunctions != null && !chatFunctions.isEmpty()) {
            builder.tools(chatFunctions); // Always include tools if available

            if (renderedFunctionCall.equalsIgnoreCase("auto")) {
                builder.toolChoice(ChatCompletionToolChoiceOption.ofAuto(ChatCompletionToolChoiceOption.Auto.AUTO));
            } else if (renderedFunctionCall.equalsIgnoreCase("none")) {
                builder.toolChoice(ChatCompletionToolChoiceOption.Auto.NONE);
            } else {
                // This is for forcing a specific function call
                // Need to ensure the requested function name exists in chatFunctions if strict
                boolean functionExists = chatFunctions.stream()
                    .map(ChatCompletionTool::function)
                    .flatMap(Optional::stream)
                    .map(f -> f.function().name())
                    .anyMatch(renderedFunctionCall::equals);

                if (!functionExists) {
                    throw new IllegalArgumentException("Requested function '" + renderedFunctionCall + "' for `functionCall` is not provided in `functions` list.");
                }

                builder.toolChoice(ChatCompletionToolChoiceOption.ofNamedToolChoice(
                    ChatCompletionNamedToolChoice.builder()
                        .function(ChatCompletionNamedToolChoice.Function.builder().name(renderedFunctionCall).build())
                        .build()
                ));
            }
        } else {
            if (renderedFunctionCall.equalsIgnoreCase("none")) {
                // If user explicitly asks for "none", we set it even without tools.
                // This tells the model not to attempt any function call.
                builder.toolChoice(ChatCompletionToolChoiceOption.Auto.NONE);
            } else if (!renderedFunctionCall.equalsIgnoreCase("auto")) {
                // If a specific function name or "required" is requested but no tools are provided, it's an error.
                // Assuming "required" is not a string value here, but handled by the specific `ofChatCompletionToolChoiceRequired()`
                // if it were to be supported via a string input.
                throw new IllegalArgumentException("Cannot specify a function name for `functionCall` ('" + renderedFunctionCall + "') when no `functions` are provided.");
            }
            // If renderedFunctionCall is "auto" (and no functions provided), we do nothing.
            // OpenAI's default for `tool_choice` is "auto" when `tools` are present,
            // and no tool_choice parameter should be sent if no `tools` are provided and "auto" is desired.
        }

        Optional.ofNullable(user).ifPresent(builder::user);
        Optional.ofNullable(stop).ifPresent(e -> builder.stop(ChatCompletionCreateParams.Stop.ofStrings(stop)));
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
            title = "Chat completion ID"
        )
        private String id;

        @Schema(
            title = "Returned object type",
            description = "Expected value is `chat.completion`."
        )
        private String object;

        @Schema(
            title = "Model used"
        )
        private String model;

        @Schema(
            title = "Completion choices",
            description = "All choices returned by the API in order."
        )
        private List<com.openai.models.chat.completions.ChatCompletion.Choice> choices;

        @Schema(
            title = "Token usage summary",
            description = "Prompt, completion, and total token counts when provided by the API."
        )
        private CompletionUsage usage;
    }

    @Builder
    @Getter
    public static class PluginChatFunctionParameter {
        @Schema(
            title = "Function parameter name"
        )
        @NotNull
        private Property<String> name;

        @Schema(
            title = "Function parameter description",
            description = "Provide enough detail for the model to populate this argument correctly."
        )
        @NotNull
        private Property<String> description;

        @Schema(
            title = "Function parameter type",
            description = "OpenAPI data type: string, number, integer, boolean, array, or object"
        )
        @NotNull
        private Property<String> type;

        @Schema(
            title = "Allowed enum values",
            description = "Optional; constrains the model to this fixed set for classification-like prompts."
        )
        private Property<List<String>> enumValues;

        @Schema(
            title = "Parameter required flag",
            description = "Defaults to false; when true the model must provide this parameter."
        )
        private Property<Boolean> required;
    }

    @Builder
    @Getter
    public static class PluginChatFunction {
        @Schema(
            title = "Function name"
        )
        private Property<String> name;

        @Schema(
            title = "Function description"
        )
        private Property<String> description;

        @Schema(
            title = "Function parameters"
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

    private static ChatCompletionCreateParams.ResponseFormat buildOpenAIResponseFormat(String schemaJson)
        throws IOException {

        @SuppressWarnings("unchecked")
        Map<String, Object> schemaMap = JacksonMapper.ofJson()
            .readValue(schemaJson, Map.class);

        var schemaBuilder = ResponseFormatJsonSchema.JsonSchema.Schema.builder();
        for (var e : schemaMap.entrySet()) {
            schemaBuilder.putAdditionalProperty(e.getKey(), JsonValue.from(e.getValue()));
        }

        var jsonSchema = ResponseFormatJsonSchema.JsonSchema.builder()
            .name("kestra_schema")
            .strict(true)
            .schema(schemaBuilder.build())
            .build();

        var rfJsonSchema = ResponseFormatJsonSchema.builder()
            .jsonSchema(jsonSchema)
            .build();

        return ChatCompletionCreateParams.ResponseFormat.ofJsonSchema(rfJsonSchema);
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

    private static FunctionDefinition toFunctionDefinition(final RunContext runContext, final PluginChatFunction function) throws RuntimeException, IllegalVariableEvaluationException {
        final Map<String, Object> functionProperties = new HashMap<>();
        final List<String> requiredList = new ArrayList<>();

        if (function.parameters != null) {
            for (PluginChatFunctionParameter parameter : runContext.render(function.parameters).asList(PluginChatFunctionParameter.class)) {
                String paramName = runContext.render(parameter.name).as(String.class).orElseThrow(() -> new IllegalVariableEvaluationException("Parameter name cannot be null"));
                String paramDescription = runContext.render(parameter.description).as(String.class).orElse(null);
                String paramType = runContext.render(parameter.type).as(String.class).orElseThrow(() -> new IllegalVariableEvaluationException("Parameter type cannot be null"));
                List<String> paramEnumValues = Optional.ofNullable(runContext.render(parameter.enumValues).asList(String.class)).orElse(Collections.emptyList());
                Boolean paramRequired = runContext.render(parameter.required).as(Boolean.class).orElse(false);

                if (paramRequired) {
                    requiredList.add(paramName);
                }

                Map<String, Object> paramSchema = new HashMap<>();
                paramSchema.put(TYPE, paramType);
                if (paramDescription != null) {
                    paramSchema.put(DESCRIPTIONS, paramDescription);
                }
                if (!paramEnumValues.isEmpty()) {
                    paramSchema.put(ENUM, paramEnumValues);
                }

                functionProperties.put(paramName, paramSchema);
            }
        }

        Map<String, Object> parametersSchema = new HashMap<>();
        parametersSchema.put(TYPE, OBJECT);
        parametersSchema.put(PROPERTIES, functionProperties);
        parametersSchema.put(REQUIRED, requiredList.isEmpty() ? Collections.emptyList() : requiredList);

        return FunctionDefinition.builder()
            .name(runContext.render(function.name).as(String.class).orElseThrow(() -> new IllegalVariableEvaluationException("Function name cannot be null")))
            .description(runContext.render(function.description).as(String.class).orElse(null))
            .parameters(JsonValue.from(parametersSchema))
            .build();
    }
}
