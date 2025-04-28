package io.kestra.plugin.openai;

import com.theokanning.openai.file.File;
import com.theokanning.openai.service.OpenAiService;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.FileUtils;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;
import lombok.experimental.SuperBuilder;

import jakarta.validation.constraints.NotNull;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;

import java.io.*;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Upload a file to OpenAI for use with other API endpoints.",
    description = "Allows uploading files that can be used with various OpenAI features, such as search, fine-tuning, or retrieval augmented responses."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Upload a file to OpenAI and get its ID for later use.",
            code = """
                id: openai_file_upload
                namespace: company.team

                inputs:
                  - id: dataFile
                    type: FILE

                tasks:
                  - id: upload_file
                    type: io.kestra.plugin.openai.UploadFile
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    file: "{{ inputs.dataFile }}"
                    purpose: "fine-tune"

                  - id: use_file_id
                    type: io.kestra.plugin.core.log.Log
                    message: "Uploaded file ID: {{ outputs.upload_file.fileId }}"
                """
        ),
        @Example(
            full = true,
            title = "Upload a user data file with description for search purposes",
            code = """
                id: openai_search_file
                namespace: company.team

                inputs:
                  - id: documentFile
                    type: FILE

                tasks:
                  - id: upload
                    type: io.kestra.plugin.openai.UploadFile
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    file: "{{ inputs.documentFile }}"
                    purpose: "search"
                    description: "Customer knowledge base articles for retrieval"
                """
        )
    }
)
public class UploadFile extends AbstractTask implements RunnableTask<UploadFile.Output> {

    @Schema(
        title = "The source file URI",
        description = "URI of the file containing data to be loaded into InfluxDB"
    )
    @NotNull
    @PluginProperty(internalStorageURI = true)
    private Property<String> from;

    @Schema(
        title = "The intended purpose of the uploaded file."
    )
    @NotNull
    private Property<String> purpose;

    @Override
    public UploadFile.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        OpenAiService client = this.client(runContext);

        String purpose = runContext.render(this.purpose).as(String.class).orElseThrow();
        String renderedFrom = runContext.render(this.from).as(String.class).orElseThrow();

        URI from = URI.create(renderedFrom);
        String extension = FileUtils.getExtension(from);

        java.io.File tempFile = runContext.workingDir().createTempFile(extension).toFile();

        Files.copy(runContext.storage().getFile(from), tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

        File fileObject = client.uploadFile(purpose,tempFile.getPath());

        return Output.builder().fileId(fileObject.getId()).build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(
            title = "The ID of the uploaded file."
        )
        private String fileId;
    }
}