package io.kestra.plugin.openai;

import com.openai.client.OpenAIClient;
import com.openai.models.Reasoning;
import com.openai.models.responses.*;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.openai.utils.ParametersUtils;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;
import lombok.experimental.SuperBuilder;

import jakarta.validation.constraints.NotNull;
import org.slf4j.Logger;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

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
            title = "Simple text input and output without tools",
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
        title = "List of input messages",
        description = "Can be a list of strings or a variable that binds to a JSON array of strings."
    )
    @NotNull
    private Object input;

    @Schema(
        title = "List of built-in tools to enable",
        description = "e.g., web_search, file_search, computer_use"
    )
    private Property<List<Tool>> tools;

    @Schema(
        title = "Controls which tool is called by the model",
        description = "none: no tools, auto: model picks, required: must use tools"
    )
    private Property<String> toolChoice = Property.of("none");

    @Schema(
        title = "Whether to persist response and chat history",
        description = "If true (default), persists in OpenAI's storage"
    )
    private Property<Boolean> store = Property.of(Boolean.TRUE);

    @Schema(
        title = "ID of previous response to continue conversation"
    )
    private Property<String> previousResponseId;

    @Schema(
        title = "Reasoning configuration",
        description = "Configuration for model reasoning process"
    )
    private Property<Reasoning> reasoning;

    @Schema(
        title = "Maximum tokens in the response"
    )
    private Property<Integer> maxOutputTokens;

    @Schema(
        title = "Sampling temperature (0-2)"
    )
    private Property<Double> temperature;

    @Schema(
        title = "Nucleus sampling parameter (0-1)"
    )
    private Property<Double> topP;

    @Schema(
        title = "Allow parallel tool execution"
    )
    private Property<Boolean> parallelToolCalls;

    @Override
    public Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        OpenAIClient client = this.openAIClient(runContext);

        String modelName = runContext.render(this.model).as(String.class).orElseThrow();
        String renderedToolChoice = runContext.render(this.toolChoice).as(String.class).orElseThrow();
        String previousResponseId = runContext.render(this.previousResponseId).as(String.class).orElse(null);
        Boolean renderedStore = runContext.render(store).as(Boolean.class).orElseThrow();

        ResponseInputItem.Message renderedInput = ParametersUtils.listParameters(runContext,input);

        ToolChoiceOptions toolChoiceOptions = ToolChoiceOptions.of(renderedToolChoice);

        Reasoning renderedReasoning = runContext.render(reasoning).as(Reasoning.class).orElseThrow();

        List<Tool> tool = runContext.render(tools).asList(Tool.class);

        ResponseInputItem responseInputItem = ResponseInputItem.ofMessage(renderedInput);

        ResponseCreateParams.Input modelInput = ResponseCreateParams.Input.ofResponse(List.of(responseInputItem));

        ResponseCreateParams createParams = ResponseCreateParams.builder()
            .input(modelInput)
            .previousResponseId(previousResponseId)
            .tools(tool)
            .toolChoice(toolChoiceOptions)
            .reasoning(renderedReasoning)
            .store(renderedStore)
            .model(modelName)
            .build();

        Response outputResponse = client.responses().create(createParams);


        List<String> sources = outputResponse.output().stream()
            .flatMap(item -> item.asMessage().content().stream())
            .findFirst()
            .map(content -> content.asOutputText().annotations())
            .orElse(Collections.emptyList())
            .stream()
            .map(ResponseOutputText.Annotation::asUrlCitation)
            .map(ResponseOutputText.Annotation.UrlCitation::url)
            .toList();

        String outputText = outputResponse.output().stream()
            .flatMap(item -> item.message().stream())
            .flatMap(message -> message.content().stream())
            .flatMap(content -> content.outputText().stream())
            .map(ResponseOutputText::text)
            .collect(Collectors.joining("\n"));

        return Output.builder()
            .responseId(outputResponse.id())
            .outputText(outputText)
            .sources(sources)
            .rawResponse(outputResponse).build();
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
        private Object rawResponse;
    }
}