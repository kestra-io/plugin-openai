package io.kestra.plugin.openai;

import com.theokanning.openai.Usage;
import com.theokanning.openai.completion.chat.*;
import com.theokanning.openai.service.OpenAiService;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;
import lombok.experimental.SuperBuilder;

import jakarta.validation.constraints.NotNull;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Given a prompt, get a response from an LLM using the [OpenAI’s Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create).",
    description = "For more information, refer to the [Chat Completions API docs](https://platform.openai.com/docs/guides/gpt/chat-completions-api)."
)
@Plugin(
    examples = {
        @Example(
            title = "Based on a prompt input, generate a completion response and pass it to a downstream task.",
            full = true,
            code = """
                id: openai
                namespace: company.team
                
                inputs:
                  - id: prompt
                    type: STRING
                    defaults: What is data orchestration?
                
                tasks:
                  - id: completion
                    type: io.kestra.plugin.openai.ChatCompletion
                    apiKey: "yourOpenAIapiKey"
                    model: gpt-4o
                    prompt: "{{ inputs.prompt }}"
                
                  - id: response
                    type: io.kestra.plugin.core.debug.Return
                    format: {{ outputs.completion.choices[0].message.content }}" 
                """
        ),
        @Example(
            title = "Based on a prompt input, ask OpenAI to call a function that determines whether you need to " +
                "respond to a customer's review immediately or wait until later, and then comes up with a " +
                "suggested response.",
            full = true,
            code = """
                id: openai
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
        title = "A list of messages comprising the conversation so far.",
        description = "Required if prompt is not set."
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
    private Property<Integer> maxTokens;

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
        OpenAiService client = this.client(runContext);

        if (this.messages == null && this.prompt == null) {
            throw new IllegalArgumentException("Either `messages` or `prompt` must be set");
        }

        List<String> stop = this.stop != null ? this.stop.asList(runContext, String.class) : null;
        String user = this.user == null ? null : this.user.as(runContext, String.class);
        String model = this.model == null ? null : this.model.as(runContext, String.class);

        List<ChatMessage> messages = new ArrayList<>();
        // Render all messages content
        if (this.messages != null) {
            for (ChatMessage message : this.messages.asList(runContext, ChatMessage.class)) {
                message.setContent(runContext.render(message.getContent()));
                messages.add(message);
            }
        }

        if (this.prompt != null) {
            messages.add(buildMessage("user", this.prompt.as(runContext, String.class)));
        }

        List<ChatFunctionDynamic> chatFunctions = null;

        if (this.functions != null) {
            chatFunctions = new ArrayList<>();
            for (PluginChatFunction function : functions.asList(runContext, PluginChatFunction.class)) {
                var chatParameters = new ChatFunctionParameters();

                if(function.parameters != null) {
                    for (PluginChatFunctionParameter parameter : function.parameters.asList(runContext, PluginChatFunctionParameter.class)) {
                        chatParameters.addProperty(ChatFunctionProperty.builder()
                            .name(parameter.name.as(runContext, String.class))
                            .description(parameter.description.as(runContext, String.class))
                            .type(parameter.type.as(runContext, String.class))
                            .required(parameter.required == null ? null : parameter.required.as(runContext, Boolean.class))
                            .enumValues(parameter.enumValues == null ? null :
                                new HashSet<>(parameter.enumValues.asList(runContext, String.class)))
                            .build()
                        );
                    }
                }

                ChatFunctionDynamic chatFunction = ChatFunctionDynamic.builder()
                    .name(function.name == null ? null : function.name.as(runContext, String.class))
                    .description(function.description == null ? null : function.description.as(runContext, String.class))
                    .parameters(chatParameters).build();

                chatFunctions.add(chatFunction);
            }
        }

        ChatCompletionRequest.ChatCompletionRequestFunctionCall chatFunctionCall = null;

        if (this.functionCall != null) {
            chatFunctionCall = ChatCompletionRequest.ChatCompletionRequestFunctionCall.of(
                this.functionCall.as(runContext, String.class)
            );
        }

        ChatCompletionResult chatCompletionResult = client.createChatCompletion(ChatCompletionRequest.builder()
            .messages(messages)
            .functions(chatFunctions)
            .functionCall(chatFunctionCall)
            .model(model)
            .temperature(this.temperature == null ? null : this.temperature.as(runContext, Double.class))
            .topP(this.topP == null ? null : this.topP.as(runContext, Double.class))
            .n(this.n == null ? null : this.n.as(runContext, Integer.class))
            .stop(stop)
            .maxTokens(this.maxTokens == null ? null : this.maxTokens.as(runContext, Integer.class))
            .presencePenalty(this.presencePenalty == null ? null : this.presencePenalty.as(runContext, Double.class))
            .frequencyPenalty(this.frequencyPenalty == null ? null : this.frequencyPenalty.as(runContext, Double.class))
            .logitBias(this.logitBias == null ? null : this.logitBias.asMap(runContext, String.class, Integer.class))
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

    @Builder
    @Getter
    public static class PluginChatFunctionParameter {
        @Schema(
            title = "The name of the function parameter."
        )
        @NotNull
        private Property<String> name;

        @Schema(
            title = "A description of the function parameter.",
            description = "Provide as many details as possible to ensure the model returns an accurate parameter."
        )
        @NotNull
        private Property<String> description;

        @Schema(
            title = "The OpenAPI data type of the parameter.",
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
            title = "Whether or not the model is required to provide this parameter.",
            description = "Defaults to false."
        )
        private Property<Boolean> required;
    }

    @Builder
    @Getter
    public static class PluginChatFunction {
        @Schema(
            title = "The name of the function."
        )
        private Property<String> name;

        @Schema(
            title = "A description of what the function does."
        )
        private Property<String> description;

        @Schema(
            title = "The function's parameters."
        )
        private Property<List<PluginChatFunctionParameter>> parameters;
    }

    private ChatMessage buildMessage(String role, String content) {
        ChatMessage message = new ChatMessage();
        message.setRole(role);
        message.setContent(content);

        return message;
    }
}
