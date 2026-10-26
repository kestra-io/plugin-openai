package io.kestra.plugin.openai;

import java.io.InterruptedIOException;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.Test;

import com.openai.core.Timeout;
import com.openai.errors.OpenAIIoException;
import com.sun.net.httpserver.HttpServer;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.utils.IdUtils;
import io.kestra.core.utils.TestsUtils;

import jakarta.inject.Inject;
import jakarta.validation.ConstraintViolationException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.spy;

@KestraTest
class AbstractTaskTest {
    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void shouldBuildClientWithConfiguredTimeout() throws Exception {
        var task = ChatCompletion.builder()
            .id(IdUtils.create())
            .type(ChatCompletion.class.getName())
            .apiKey(Property.ofValue("sk-mock-key"))
            .model(Property.ofValue("gpt-4o"))
            .clientTimeout(Property.ofValue(45L))
            .build();
        var runContext = TestsUtils.mockRunContext(runContextFactory, task, Map.of());

        var ref = new AtomicReference<Timeout>();
        task.openAIClient(runContext).withOptions(b -> ref.set(b.timeout()));
        assertThat(ref.get().request()).isEqualTo(Duration.ofSeconds(45));
    }

    @Test
    void shouldDefaultClientTimeoutTo600Seconds() throws Exception {
        var task = ChatCompletion.builder()
            .id(IdUtils.create())
            .type(ChatCompletion.class.getName())
            .apiKey(Property.ofValue("sk-mock-key"))
            .model(Property.ofValue("gpt-4o"))
            .build();
        var runContext = TestsUtils.mockRunContext(runContextFactory, task, Map.of());

        var ref = new AtomicReference<Timeout>();
        task.openAIClient(runContext).withOptions(b -> ref.set(b.timeout()));
        assertThat(ref.get().request()).isEqualTo(Duration.ofSeconds(600));
    }

    @Test
    void shouldRejectClientTimeoutBelowOne() {
        var task = ChatCompletion.builder()
            .id(IdUtils.create())
            .type(ChatCompletion.class.getName())
            .apiKey(Property.ofValue("sk-mock-key"))
            .model(Property.ofValue("gpt-4o"))
            .clientTimeout(Property.ofValue(0L))
            .build();
        var runContext = TestsUtils.mockRunContext(runContextFactory, task, Map.of());

        assertThatThrownBy(() -> task.openAIClient(runContext))
            .isInstanceOf(ConstraintViolationException.class)
            .hasMessageContaining("clientTimeout");
    }

    @Test
    void shouldThrowTimeoutWhenResponseExceedsClientTimeout() throws Exception {
        var server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/chat/completions", exchange ->
        {
            try {
                Thread.sleep(2500);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            exchange.sendResponseHeaders(200, 0);
            exchange.close();
        });
        server.start();

        try {
            var runContext = runContextFactory.of();
            var rawTask = ChatCompletion.builder()
                .apiKey(Property.ofValue("sk-mock-key"))
                .clientTimeout(Property.ofValue(1L))
                .model(Property.ofValue("gpt-4o"))
                .messages(
                    Property.ofValue(
                        List.of(
                            ChatCompletion.ChatMessage.builder()
                                .role("user")
                                .content("hello")
                                .build()
                        )
                    )
                )
                .build();

            var client = rawTask.openAIClient(runContext)
                .withOptions(b -> b.baseUrl("http://localhost:%d/".formatted(server.getAddress().getPort())).maxRetries(0));

            var task = spy(rawTask);
            doReturn(client).when(task).openAIClient(any(RunContext.class));

            assertThatThrownBy(() -> task.run(runContext))
                .isInstanceOf(OpenAIIoException.class)
                .hasCauseInstanceOf(InterruptedIOException.class);
        } finally {
            server.stop(0);
        }
    }
}
