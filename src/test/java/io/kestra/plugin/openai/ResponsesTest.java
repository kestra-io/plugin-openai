package io.kestra.plugin.openai;

import com.openai.core.JsonValue;
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
            .apiKey(Property.ofValue(getApiKey()))
            .clientTimeout(30)
            .model(Property.ofValue("gpt-4.1"))
            .input(Property.ofValue("Explain what Kestra is in one sentence."))
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

        Map<String, Object> webSearchTool = Map.of(
            "type", "web_search_preview",
             "search_context_size", "low",
             "user_location", Map.of("type", "approximate", "city", "Berlin", "region", "Berlin", "country", "DE")
        );

        Responses task = Responses.builder()
            .id("test-web-search")
            .type(Responses.class.getName())
            .apiKey(Property.ofValue(getApiKey()))
            .clientTimeout(30)
            .model(io.kestra.core.models.property.Property.ofValue("gpt-4.1"))
            .input(Property.ofValue("What is the latest version of Kestra?"))
            .tools(io.kestra.core.models.property.Property.ofValue(Collections.singletonList(webSearchTool)))
            .toolChoice(io.kestra.core.models.property.Property.ofValue(Responses.ToolChoice.REQUIRED))
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

        Map<String, Object> functionTool = Map.of(
            "type", "function",
            "name", "respond_to_review",
            "description", "Analyze the sentiment of the provided text in one word",
            "strict", true,
            "parameters", toolParameters
        );

        Responses task = Responses.builder()
            .id("test-function")
            .type(Responses.class.getName())
            .apiKey(Property.ofValue(getApiKey()))
            .model(io.kestra.core.models.property.Property.ofValue("gpt-4.1"))
            .input(Property.ofValue("I'm so happy about this new feature!"))
            .tools(io.kestra.core.models.property.Property.ofValue(Collections.singletonList(functionTool)))
            .toolChoice(io.kestra.core.models.property.Property.ofValue(Responses.ToolChoice.REQUIRED))
            .build();

        Responses.Output output = task.run(runContext);

        assertNotNull(output);
        assertNotNull(output.getOutputText());
        assertThat(output.getOutputText(), anyOf(
            containsStringIgnoringCase("happy"),
            containsStringIgnoringCase("enjoy"),
            containsStringIgnoringCase("positive")
        ));
    }
}