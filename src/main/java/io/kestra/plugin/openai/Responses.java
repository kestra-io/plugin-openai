package io.kestra.plugin.openai;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.openai.client.OpenAIClient;
import com.openai.models.responses.*;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.executions.metrics.Counter;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.openai.utils.ParametersUtils;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Call OpenAI Responses with tools",
    description = "Uses the Responses API for chat, tool calls, and structured text output. Supports web/file search and function tools, optional JSON Schema formatting, continuation via `previousResponseId`, and persistence enabled by default. See the [Responses docs](https://platform.openai.com/docs/guides/responses)."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Send a simple text prompt to OpenAI and output the result as text.",
            code = """
                id: simple_text
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: Explain what is Kestra in 3 sentences

                tasks:
                  - id: explain
                    type: io.kestra.plugin.openai.Responses
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    model: gpt-4.1-mini
                    input: "{{ inputs.prompt }}"

                  - id: log
                    type: io.kestra.plugin.core.log.Log
                    message: "{{ outputs.explain.outputText }}"
                """
        ),
        @Example(
            full = true,
            title = "Use the OpenAI's web-search tool to find recent trends in workflow orchestration.",
            code = """
                id: web_search
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: List recent trends in workflow orchestration

                tasks:
                  - id: trends
                    type: io.kestra.plugin.openai.Responses
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    model: gpt-4.1-mini
                    input: "{{ inputs.prompt }}"
                    toolChoice: REQUIRED
                    tools:
                      - type: web_search_preview

                  - id: log
                    type: io.kestra.plugin.core.log.Log
                    message: "{{ outputs.trends.outputText }}"
                """
        ),
        @Example(
            full = true,
            title = "Use the OpenAI's web-search tool to get a daily summary of local news via email.",
            code = """
                id: fetch_local_news
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: Summarize top 5 news from my region

                tasks:
                  - id: news
                    type: io.kestra.plugin.openai.Responses
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    model: gpt-4.1-mini
                    input: "{{ inputs.prompt }}"
                    toolChoice: REQUIRED
                    tools:
                      - type: web_search_preview
                        search_context_size: low  # optional; low, medium, high
                        user_location:
                          type: approximate # OpenAI doesn't provide other types atm, and it cannot be omitted
                          city: Berlin
                          region: Berlin
                          country: DE

                  - id: mail
                    type: io.kestra.plugin.notifications.mail.MailSend
                    from: your_email
                    to: your_email
                    username: your_email
                    host: mail.privateemail.com
                    port: 465
                    password: "{{ secret('EMAIL_PASSWORD') }}"
                    sessionTimeout: 6000
                    subject: Daily News Summary
                    htmlTextContent: "{{ outputs.news.outputText }}"

                triggers:
                  - id: schedule
                    type: io.kestra.plugin.core.trigger.Schedule
                    cron: "0 9 * * *"
                """
        ),
        @Example(
            full = true,
            title = "Use the OpenAI's function-calling tool to respond to a customer review and determine urgency of response.",
            code = """
                id: responses_functions
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: I love your product and would purchase it again!

                tasks:
                  - id: openai
                    type: io.kestra.plugin.openai.Responses
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    model: gpt-4.1-mini
                    input: "{{ inputs.prompt }}"
                    toolChoice: AUTO
                    tools:
                      - type: function
                        name: respond_to_review
                        description: >-
                          Given the customer product review provided as input, determine how
                          urgently a reply is required and then provide suggested response text.
                        strict: true
                        parameters:
                          type: object
                          required:
                            - response_urgency
                            - response_text
                          properties:
                            response_urgency:
                              type: string
                              description: >-
                                How urgently this customer review needs a reply. Bad reviews must
                                be addressed immediately before anyone sees them. Good reviews
                                can wait until later.
                              enum:
                                - reply_immediately
                                - reply_later
                            response_text:
                              type: string
                              description: The text to post online in response to this review.
                          additionalProperties: false

                  - id: output
                    type: io.kestra.plugin.core.output.OutputValues
                    values:
                      urgency: "{{ fromJson(outputs.openai.outputText).response_urgency }}"
                      response: "{{ fromJson(outputs.openai.outputText).response_text }}"
                """
        ),
        @Example(
            full = true,
            title = "Run a stateful chat with OpenAI using the Responses API.",
            code = """
                id: stateful_chat
                namespace: company.team

                inputs:
                  - id: user_input
                    type: STRING
                    defaults: How can I get started with Kestra as a microservice developer?

                  - id: reset_conversation
                    type: BOOL
                    defaults: false

                tasks:
                  - id: maybe_reset_conversation
                    runIf: "{{ inputs.reset_conversation }}"
                    type: io.kestra.plugin.core.kv.Delete
                    key: "RESPONSE_ID"

                  - id: chat_request
                    type: io.kestra.plugin.openai.Responses
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    model: gpt-4.1
                    previousResponseId: "{{ kv('RESPONSE_ID', errorOnMissing=false) }}"
                    input:
                      - role: user
                        content:
                          - type: input_text
                            text: "{{ inputs.user_input }}"

                  - id: store_response
                    type: io.kestra.plugin.core.kv.Set
                    key: "RESPONSE_ID"
                    value: "{{ outputs.chat_request.responseId }}"

                  - id: output_log
                    type: io.kestra.plugin.core.log.Log
                    message: "Response: {{ outputs.chat_request.outputText }}"
                """
        ),
        @Example(
            full = true,
            title = "Return a structured output with nutritional information about a food item using OpenAI's Responses API.",
            code = """
                id: structured_output_demo
                namespace: company.team

                inputs:
                  - id: food
                    type: STRING
                    defaults: Avocado

                tasks:
                  - id: generate_structured_response
                    type: io.kestra.plugin.openai.Responses
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    model: gpt-4.1-mini
                    input: "Fill in nutrients information for the following food: {{ inputs.food }}"
                    text:
                      format:
                        type: json_schema
                        name: food_macronutrients
                        schema:
                          type: object
                          properties:
                            food:
                              type: string
                              description: The name of the food or meal.
                            macronutrients:
                              type: object
                              description: Macro-nutritional content of the food.
                              properties:
                                carbohydrates:
                                  type: number
                                  description: Amount of carbohydrates in grams.
                                proteins:
                                  type: number
                                  description: Amount of proteins in grams.
                                fats:
                                  type: number
                                  description: Amount of fats in grams.
                              required:
                                - carbohydrates
                                - proteins
                                - fats
                              additionalProperties: false
                            vitamins:
                              type: object
                              description: Specific vitamins present in the food.
                              properties:
                                vitamin_a:
                                  type: number
                                  description: Amount of Vitamin A in micrograms.
                                vitamin_c:
                                  type: number
                                  description: Amount of Vitamin C in milligrams.
                                vitamin_d:
                                  type: number
                                  description: Amount of Vitamin D in micrograms.
                                vitamin_e:
                                  type: number
                                  description: Amount of Vitamin E in milligrams.
                                vitamin_k:
                                  type: number
                                  description: Amount of Vitamin K in micrograms.
                              required:
                                - vitamin_a
                                - vitamin_c
                                - vitamin_d
                                - vitamin_e
                                - vitamin_k
                              additionalProperties: false
                          required:
                            - food
                """

        )
    }
)
public class Responses extends AbstractTask implements RunnableTask<Responses.Output> {

    @Schema(
        title = "Model ID",
        description = "Required OpenAI model (e.g., gpt-4.1 or gpt-4o); see the [model docs](https://platform.openai.com/docs/models)."
    )
    @NotNull
    private Property<String> model;

    @Schema(
        title = "Input payload",
        description = "String or structured `input` list for the conversation; required."
    )
    @NotNull
    private Property<Object> input;

    @Schema(
        title = "Text response config",
        description = "Optional map for text output settings (e.g., `json_schema` formatted responses)."
    )
    private Property<Map<String, Object>> text;

    @Schema(
        title = "Enabled tools",
        description = "List of tool objects (web_search_preview, file_search, function, etc.) sent to the API."
    )
    private Property<List<Map<String, Object>>> tools;

    @Schema(
        title = "Tool choice",
        description = "NONE disables tools, AUTO (default) lets the model decide, REQUIRED forces a tool call when tools are provided."
    )
    private Property<ToolChoice> toolChoice;

    @Schema(
        title = "Persist response history",
        description = "Defaults to true to store conversation in OpenAI; set false for ephemeral exchanges."
    )
    @NotNull
    @Builder.Default
    private Property<Boolean> store = Property.ofValue(true);

    @Schema(
        title = "Previous response ID",
        description = "Continue a conversation by supplying a prior `response_id`; requires prior persistence."
    )
    private Property<String> previousResponseId;

    @Schema(
        title = "Reasoning configuration",
        description = "Optional reasoning options map passed to the API."
    )
    private Property<Map<String, String>> reasoning;

    @Schema(
        title = "Max output tokens",
        description = "Caps response tokens; leave unset to use OpenAI defaults."
    )
    private Property<Integer> maxOutputTokens;

    @Schema(
        title = "Sampling temperature (0-2)",
        description = "Default 1.0; higher values increase randomness."
    )
    @Builder.Default
    private Property<@Max(2)Double> temperature = Property.ofValue(1.0);

    @Schema(
        title = "Top-p nucleus sampling (0-1)",
        description = "Default 1.0; lower values limit candidate tokens."
    )
    @Builder.Default
    private Property<@Max(1) Double> topP = Property.ofValue(1.0);

    @Schema(
        title = "Parallel tool calls",
        description = "Whether tools may run in parallel; uses provider default when unset."
    )
    private Property<Boolean> parallelToolCalls;

    @Override
    public Output run(RunContext runContext) throws Exception {
        OpenAIClient client = this.openAIClient(runContext);

        String modelName = runContext.render(this.model).as(String.class).orElseThrow();
        ToolChoice renderedToolChoice = runContext.render(this.toolChoice).as(ToolChoice.class).orElse(ToolChoice.AUTO);
        Boolean renderedStore = runContext.render(store).as(Boolean.class).orElse(Boolean.TRUE);
        String renderedPreviousResponseId = runContext.render(this.previousResponseId).as(String.class).orElse(null);
        Integer maxTokens = runContext.render(this.maxOutputTokens).as(Integer.class).orElse(null);
        Double renderedTemperature = runContext.render(this.temperature).as(Double.class).orElse(1.0);
        Double renderedTopP = runContext.render(this.topP).as(Double.class).orElse(1.0);
        Boolean parallelCalls = runContext.render(this.parallelToolCalls).as(Boolean.class).orElse(null);

        Map<String, String> renderedReasoningMap = runContext.render(reasoning).asMap(String.class, String.class);
        Map<String, Object> renderedTextFormat = runContext.render(text).asMap(String.class, Object.class);

        List<ResponseInputItem.Message> renderedInput = ParametersUtils.listParameters(runContext, input);

        List<Map<String, Object>> renderedTools = runContext.render(this.tools).asList(Map.class);

        List<ResponseInputItem> responseInputItem = renderedInput.stream().map(ResponseInputItem::ofMessage).toList();

        ResponseCreateParams.Input modelInput = ResponseCreateParams.Input.ofResponse(responseInputItem);

        ResponseCreateParams.Builder paramsBuilder = ResponseCreateParams.builder()
            .input(modelInput)
            .model(modelName)
            .store(renderedStore)
            .temperature(renderedTemperature)
            .topP(renderedTopP);

        if (renderedTools != null && !renderedTools.isEmpty()) {
            List<Tool> sdkTools = renderedTools.stream()
                .map(t -> ParametersUtils.OBJECT_MAPPER.convertValue(t, com.openai.models.responses.Tool.class))
                .toList();

            paramsBuilder.tools(sdkTools);
        }

        paramsBuilder.toolChoice(ToolChoiceOptions.of(renderedToolChoice.name().toLowerCase(Locale.ROOT)));

        if (renderedPreviousResponseId != null && !renderedPreviousResponseId.isEmpty()) {
            paramsBuilder.previousResponseId(renderedPreviousResponseId);
        }

        if (renderedReasoningMap != null) {
            com.openai.models.Reasoning renderedReasoning = ParametersUtils.OBJECT_MAPPER.convertValue(renderedReasoningMap,
                com.openai.models.Reasoning.class);
            paramsBuilder.reasoning(renderedReasoning);
        }

        if (maxTokens != null) {
            paramsBuilder.maxOutputTokens(maxTokens);
        }

        if (parallelCalls != null) {
            paramsBuilder.parallelToolCalls(parallelCalls);
        }

        if (renderedTextFormat != null) {
            ResponseTextConfig textFormat = ParametersUtils.OBJECT_MAPPER.convertValue(renderedTextFormat, ResponseTextConfig.class);
            paramsBuilder.text(textFormat);
        }

        ResponseCreateParams params = paramsBuilder.build();

        Response outputResponse = client.responses().create(params);

        if (outputResponse.usage().isPresent()) {
            runContext.metric(Counter.of("usage.prompt.tokens", outputResponse.usage().get().inputTokens()));
            runContext.metric(Counter.of("usage.completion.tokens", outputResponse.usage().get().outputTokens()));
            runContext.metric(Counter.of("usage.total.tokens", outputResponse.usage().get().totalTokens()));
        }

        List<String> sources = outputResponse.output().stream()
            .flatMap(item -> item.message().stream())
            .flatMap(message -> message.content().stream())
            .flatMap(content -> content.outputText().stream())
            .flatMap(outputText -> outputText.annotations().stream())
            .map(ResponseOutputText.Annotation::asUrlCitation)
            .filter(Objects::nonNull)
            .map(ResponseOutputText.Annotation.UrlCitation::url)
            .toList();

        String outputText = ParametersUtils.extractOutputText(outputResponse.output());

        return Output.builder()
            .responseId(outputResponse.id())
            .outputText(outputText)
            .sources(sources)
            .rawResponse(JacksonMapper.toMap(outputResponse))
            .build();
    }

    public enum ToolChoice {
        NONE,
        AUTO,
        REQUIRED
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(
            title = "Response ID"
        )
        private String responseId;

        @Schema(
            title = "Generated text"
        )
        private String outputText;

        @Schema(
            title = "Sources",
            description = "URLs returned via web/file search annotations."
        )
        private List<String> sources;

        @Schema(
            title = "Raw API response",
            description = "Full response as a map for downstream processing."
        )
        private Map<String, Object> rawResponse;
    }
}
