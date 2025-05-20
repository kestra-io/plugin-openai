package io.kestra.plugin.openai;

import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.storages.StorageInterface;
import io.kestra.core.tenant.TenantService;
import io.kestra.core.utils.IdUtils;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledIf;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Objects;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.notNullValue;


@KestraTest
@DisabledIf(
    value = "canNotBeEnabled",
    disabledReason = "Needs an OpenAI API Key to work"
)
class UploadFileTest extends AbstractOpenAITest {
    @Inject
    private RunContextFactory runContextFactory;

    @Inject
    protected StorageInterface storageInterface;

    @Test
    void run() throws Exception {
        URI source = storagePut("1.yml");

        RunContext runContext = runContextFactory.of();

        UploadFile task = UploadFile.builder()
            .apiKey(Property.of(getApiKey()))
            .from(Property.of(source.toString()))
            .purpose(Property.of("user_data"))
            .build();

        UploadFile.Output runOutput = task.run(runContext);

        assertThat(runOutput.getFileId(), notNullValue());
    }

    protected URI storagePut(String path) throws URISyntaxException, IOException {
        return storageInterface.put(
            TenantService.MAIN_TENANT,
            null,
            new URI("/" + (path != null ? path : IdUtils.create())),
            new FileInputStream(file())
        );
    }

    protected static File file() throws URISyntaxException {
        return new File(Objects.requireNonNull(AbstractTask.class.getClassLoader()
                .getResource("application.yml"))
            .toURI());
    }
}
