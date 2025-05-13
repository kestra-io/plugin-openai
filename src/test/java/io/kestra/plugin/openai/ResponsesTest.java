package io.kestra.plugin.openai;

import com.openai.core.JsonValue;
import com.openai.models.responses.FunctionTool;
import com.openai.models.responses.Tool;
import com.openai.models.responses.WebSearchTool;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledIf;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@KestraTest
@DisabledIf(
    value = "canNotBeEnabled",
    disabledReason = "Needs an OpenAI API Key to work"
)
public class ResponsesTest extends AbstractOpenAITest {

    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void testSimplePrompt() throws Exception {
        RunContext runContext = runContextFactory.of();

        Responses task = Responses.builder()
            .id("test-task")
            .type(Responses.class.getName())
            .apiKey(Property.of(getApiKey()))
            .clientTimeout(30)
            .model(Property.of("gpt-4.1"))
            .input(Property.of("Explain what Kestra is in one sentence."))
            .build();


        Responses.Output output = task.run(runContext);

        assertNotNull(output);
        assertNotNull(output.getResponseId());
        assertNotNull(output.getOutputText());
        assertThat(output.getOutputText(), containsString("Kestra"));
        assertNotNull(output.getRawResponse());
    }

    @Test
    void testWebSearchTool() throws Exception {
        RunContext runContext = runContextFactory.of();

        Tool webSearchTool = Tool.ofWebSearch(WebSearchTool.builder()
            .type(WebSearchTool.Type.WEB_SEARCH_PREVIEW)
            .build());

        Responses task = Responses.builder()
            .id("test-web-search")
            .type(Responses.class.getName())
            .apiKey(Property.of(getApiKey()))
            .clientTimeout(30)
            .model(io.kestra.core.models.property.Property.of("gpt-4.1"))
            .input(Property.of("What is the latest version of Kestra?"))
            .tools(io.kestra.core.models.property.Property.of(Collections.singletonList(webSearchTool)))
            .toolChoice(io.kestra.core.models.property.Property.of(Responses.ToolChoice.REQUIRED))
            .build();

        Responses.Output output = task.run(runContext);

        assertNotNull(output);
        assertNotNull(output.getOutputText());
        assertThat(output.getSources(), is(not(empty())));
    }

    @Test
    void testFunctionTool() throws Exception {
        RunContext runContext = runContextFactory.of();

        Map<String, JsonValue> toolParameters = Map.of(
            "type", JsonValue.from("object"),
            "required", JsonValue.from(List.of("response_urgency", "response_text")),
            "properties", JsonValue.from(Map.of(
                "response_urgency", Map.of(
                    "type", "string",
                    "description", "How urgently this customer review needs a reply. Bad reviews must be addressed immediately before anyone sees them. Good reviews can wait until later.",
                    "enum", List.of("reply_immediately", "reply_later")
                ),
                "response_text", Map.of(
                    "type", "string",
                    "description", "The text to post online in response to this review."
                )
            )),
            "additionalProperties", JsonValue.from(false)
        );

        Tool functionTool = Tool.ofFunction(FunctionTool.builder()
            .name("respond_to_review")
            .description("Analyze the sentiment of the provided text")
            .strict(true)
            .parameters(FunctionTool.Parameters.builder().additionalProperties(toolParameters).build())
            .build());

        Responses task = Responses.builder()
            .id("test-function")
            .type(Responses.class.getName())
            .apiKey(Property.of(getApiKey()))
            .model(io.kestra.core.models.property.Property.of("gpt-4.1"))
            .input(Property.of("I'm so happy about this new feature!"))
            .tools(io.kestra.core.models.property.Property.of(Collections.singletonList(functionTool)))
            .toolChoice(io.kestra.core.models.property.Property.of(Responses.ToolChoice.REQUIRED))
            .build();

        Responses.Output output = task.run(runContext);

        assertNotNull(output);
        assertNotNull(output.getOutputText());
        assertThat(output.getOutputText(), containsString("positive"));
    }
}