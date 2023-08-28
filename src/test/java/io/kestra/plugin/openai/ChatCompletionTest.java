package io.kestra.plugin.openai;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import com.theokanning.openai.completion.chat.ChatFunctionCall;
import com.theokanning.openai.completion.chat.ChatMessage;
import com.theokanning.openai.service.OpenAiService;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.micronaut.test.extensions.junit5.annotation.MicronautTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;


@MicronautTest
@Disabled("Needs an OpenAI API Key to work")
public class ChatCompletionTest {
    @Inject
    private RunContextFactory runContextFactory;

    private String apiKey = "";

    @Test
    void runMessages() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatMessage> messages = List.of(
            buildMessage("user","what is the capital of France?")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(this.apiKey)
            .model("gpt-3.5-turbo-0613")
            .messages(messages)
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().get(0).getMessage().getContent(), containsString("Paris"));
        assertThat(runOutput.getModel(), containsString("gpt-3.5-turbo"));
        assertThat(runOutput.getUsage().getPromptTokens(), is(14L));
    }

    @Test
    void runPrompt() throws Exception {
        RunContext runContext = runContextFactory.of();

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(this.apiKey)
            .model("gpt-3.5-turbo-0613")
            .prompt("what is the capital of France?")
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().get(0).getMessage().getContent(), containsString("Paris"));
        assertThat(runOutput.getModel(), containsString("gpt-3.5-turbo"));
        assertThat(runOutput.getUsage().getPromptTokens(), is(14L));
    }

    @Test
    void runMessagesWithPrompt() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatMessage> messages = List.of(
            buildMessage("user","what is the capital of France?"),
            buildMessage("assistant","The capital of France is Paris.")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(this.apiKey)
            .model("gpt-3.5-turbo-0613")
            .messages(messages)
            .prompt("and the capital of germany?")
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().get(0).getMessage().getContent(), containsString("Berlin"));
        assertThat(runOutput.getModel(), containsString("gpt-3.5-turbo"));
        assertThat(runOutput.getUsage().getPromptTokens(), is(35L));
    }

    @Test
    void runFunctionWithMessage() throws Exception {
        RunContext runContext = runContextFactory.of();

        String openAiFunctionSchema = "{\n" +
            "  \"type\": \"object\",\n" +
            "  \"properties\": {\n" +
            "    \"location\": {\n" +
            "      \"type\": \"string\",\n" +
            "      \"description\": \"The city and state/province, and country, e.g. San Francisco, CA, USA\"\n" +
            "    },\n" +
            "    \"unit\": {\n" +
            "      \"type\": \"string\",\n" +
            "      \"description\": \"The temperature unit this city uses.\",\n" +
            "      \"enum\": [\n" +
            "        \"celsius\",\n" +
            "        \"fahrenheit\"\n" +
            "      ]\n" +
            "    }\n" +
            "  },\n" +
            "  \"required\": [\n" +
            "    \"location\",\n" +
            "    \"unit\"\n" +
            "  ]\n" +
            "}";

        ObjectMapper mapper = OpenAiService.defaultObjectMapper();

        JsonParser jp = mapper.getFactory().createParser(openAiFunctionSchema);
        JsonNode parameters = mapper.readTree(jp);

        List<ChatCompletion.PluginChatFunction> functions = List.of(
            new ChatCompletion.PluginChatFunction("test", "finds the most relevant city and its temperature unit", parameters)
        );

        List<ChatMessage> messages = List.of(
            buildMessage("user","I was travelling along the Mediterranean coast, and I ended up in Lyon.")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(this.apiKey)
            .model("gpt-3.5-turbo-0613")
            .apiTimeout(30)
            .messages(messages)
            .functions(functions)
            .functionCall("test")
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().get(0).getFinishReason(), containsString("function_call"));

        ChatFunctionCall functionCall = runOutput.getChoices().get(0).getMessage().getFunctionCall();
        assertThat(functionCall.getName(), containsString("test"));
        assertThat(functionCall.getArguments().get("location").toString(), containsString("Lyon"));

        assertThat(runOutput.getModel(), containsString("gpt-3.5-turbo"));
    }

    private ChatMessage buildMessage(String role, String content) {
        ChatMessage message = new ChatMessage();
        message.setRole(role);
        message.setContent(content);

        return message;
    }
}
