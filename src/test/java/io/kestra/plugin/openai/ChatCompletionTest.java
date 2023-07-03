package io.kestra.plugin.openai;

import com.theokanning.openai.completion.chat.ChatMessage;
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

    private ChatMessage buildMessage(String role, String content) {
        ChatMessage message = new ChatMessage();
        message.setRole(role);
        message.setContent(content);

        return message;
    }
}
