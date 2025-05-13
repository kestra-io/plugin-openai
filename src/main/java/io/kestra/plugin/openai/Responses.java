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
    title = "Create responses using OpenAI's new Responses API with built-in tools and structured outputs.",
    description = "For more information, refer to the [OpenAI Responses API docs](https://platform.openai.com/docs/guides/responses)."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Simple text input and output.",
            code = """
                id: responses_text
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: Explain what is Kestra in 3 sentences

                tasks:
                  - id: explain
                    type: io.kestra.plugin.openai.Responses
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    model: gpt-4.1
                    input: "{{ inputs.prompt }}"

                  - id: log
                    type: io.kestra.plugin.core.log.Log
                    message: "{{ outputs.explain.outputText }}"
            """
        ),
        @Example(
            full = true,
            title = "Structured output with web search tool",
            code = """
                id: responses_json_search
                namespace: company.team

                inputs:
                  - id: prompt
                    type: STRING
                    defaults: "List recent trends in workflow orchestration. Return as JSON."

                tasks:
                  - id: trends
                    type: io.kestra.plugin.openai.Responses
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    model: gpt-4.1
                    input: "{{ inputs.prompt }}"
                    text:
                      format:
                        type: json_object
                    toolChoice: required
                    tools:
                      - type: web_search_preview

                  - id: log
                    type: io.kestra.plugin.core.log.Log
                    message: "{{ outputs.trends.outputText }}"
            """
        )
    }
)
public class Responses extends AbstractTask implements RunnableTask<Responses.Output> {

    @Schema(
        title = "Model name to use (e.g., gpt-4o)",
        description = "See the [OpenAI model's documentation](https://platform.openai.com/docs/models)"
    )
    @NotNull
    private Property<String> model;

    @Schema(
        title = "Input for the model"
    )
    @NotNull
    private Property<Object> input;

    @Schema(
        title = "Text response configuration",
        description = "Configure the format of the model's text response"
    )
    private Property<Map<String, Object>> text;

    @Schema(
        title = "List of built-in tools to enable",
        description = "e.g., web_search, file_search, function calling"
    )
    private Property<List<Tool>> tools;

    @Schema(
        title = "Controls which tool is called by the model",
        description = "none: no tools, auto: model picks, required: must use tools"
    )
    private Property<ToolChoice> toolChoice;

    @Schema(
        title = "Whether to persist response and chat history",
        description = "If true (default), persists in OpenAI's storage"
    )
    @NotNull
    @Builder.Default
    private Property<Boolean> store = Property.of(true);

    @Schema(
        title = "ID of previous response to continue conversation"
    )
    private Property<String> previousResponseId;

    @Schema(
        title = "Reasoning configuration",
        description = "Configuration for model reasoning process"
    )
    private Property<Map<String, String>> reasoning;

    @Schema(
        title = "Maximum tokens in the response"
    )
    private Property<Integer> maxOutputTokens;

    @Schema(
        title = "Sampling temperature (0-2)"
    )
    @Builder.Default
    private Property<@Max(2)Double> temperature = Property.of(1.0);

    @Schema(
        title = "Nucleus sampling parameter (0-1)"
    )
    @Builder.Default
    private Property<@Max(1) Double> topP = Property.of(1.0);

    @Schema(
        title = "Allow parallel tool execution"
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

        List<Tool> renderedTools = runContext.render(tools).asList(Tool.class);

        List<ResponseInputItem> responseInputItem = renderedInput.stream().map(ResponseInputItem::ofMessage).toList();

        ResponseCreateParams.Input modelInput = ResponseCreateParams.Input.ofResponse(responseInputItem);

        ResponseCreateParams.Builder paramsBuilder = ResponseCreateParams.builder()
            .input(modelInput)
            .model(modelName)
            .store(renderedStore)
            .temperature(renderedTemperature)
            .topP(renderedTopP);

        if (renderedTools != null && !renderedTools.isEmpty()) {
            paramsBuilder.tools(renderedTools);
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
            title = "ID of the persisted response"
        )
        private String responseId;

        @Schema(
            title = "The generated text output"
        )
        private String outputText;

        @Schema(
            title = "List of sources (for web/file search)"
        )
        private List<String> sources;

        @Schema(
            title = "Full API response for advanced use"
        )
        private Map<String, Object> rawResponse;
    }
}